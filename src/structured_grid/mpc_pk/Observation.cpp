#include <winstd.H>

#include <PMAmr.H>
#include <Observation.H>
#include <EventCoord.H>
#include <PMAMR_Labels.H>


PMAmr* Observation::amrp;
std::map<std::string, int> Observation::obs_type_list;

static bool initialized = false;

void
Observation::Cleanup ()
{
    Observation::amrp = 0;
    Observation::obs_type_list.clear();
    initialized = false;
}

void
Observation::Initialize()
{
    BoxLib::ExecOnFinalize(Observation::Cleanup);
    initialized = true;
    
    obs_type_list["average"]          = 0;
    obs_type_list["integral"]         = 1;
    obs_type_list["time_integral"]    = 2;
    obs_type_list["squared_integral"] = 3;
    obs_type_list["flux"]             = 4;
    obs_type_list["point_sample"]     = 5;
    obs_type_list["peak_value"]       = 6;
}


Observation::Observation(const std::string& name,
                         const std::string& field,
                         const Region&      region,
                         const std::string& obs_type,
                         const std::string& event_label)
    : name(name), field(field), region(region), obs_type(obs_type), 
      event_label(event_label), obs_data_initialized(false)
{
    if (!initialized) {
        Initialize();
    }

    if (obs_type_list.find(obs_type) == obs_type_list.end()) {
        BoxLib::Abort("Unsupported observation type");
    }
}

std::pair<bool,Real>
process_events(Real time, Real dt, int iter, int diter, const std::string& event_label)
{
    if (Observation::PMAmrPtr() == 0) {
        BoxLib::Abort("Observations not yet associated with a PMAmr");
    }
    EventCoord& event_coord = Observation::PMAmrPtr()->eventCoord();
    std::pair<Real,Array<std::string> > nextEvent = event_coord.NextEvent(time,dt,iter, diter);
    if (nextEvent.second.size()) 
    {
        // Process event
        const Array<std::string>& eventList = nextEvent.second;

        for (int j=0; j<eventList.size(); ++j) {
            if (event_label == eventList[j]) {
                return std::pair<bool,Real>(true,nextEvent.first);
            }
        }
    }
    return std::pair<bool,Real>(false,-1);
}


void 
Observation::process(Real t_old, 
		     Real t_new,
                     int  iter,
		     int  verbose)
{
  // Must set the amr pointer prior to use via Observation::setAmrPtr
  BL_ASSERT(amrp); 

  // determine observation at time t_new
  switch (obs_type_list[obs_type] )
    {
    case 0:
      val_new  = average(t_new);

      break;
      
    case 1:
      val_new = volume_integral(t_new);
      break;
      
    case 2:
      // times[nObs] = 0.5*(t_old+t_new);
      val_new = volume_time_integral(t_old,t_new);
      break;

    case 5:
      val_new = point_sample(t_new);
      break;

    case 6:
      val_new = peak_sample(t_new);
      break;

    default:
      ;// Do nothing
    }

  if (!obs_data_initialized) {
      val_old = val_new;
      obs_data_initialized = true;
  }

  // determine which of the observations are requested at this time step
  switch (obs_type_list[obs_type] )
    {
    case 2:
      if (vals.size() < 1) 
	{
	  vals[0] = 0.;
	}
      if (times[0] <= t_old && times[1] >= t_new)
	vals[0] = vals[0] + val_new;   
      break;

    default:

        std::pair<bool,Real> ret = process_events(t_old,t_new-t_old,iter,1,event_label);
        
        if (ret.first) {
            Real dt_red = ret.second < 0 ? t_new - t_old : ret.second;
            times.push_back(t_old + dt_red);
            Real eta = std::min(1.,std::max(0.,dt_red/(t_new-t_old)));
            vals[times.size()-1] = (val_old*(1 - eta) + val_new*eta);

	    if (verbose>0 && ParallelDescriptor::IOProcessor()) {
	      const int old_prec = std::cout.precision(16);
	      std::cout.setf(std::ios::scientific);
	      std::cout << "Observation::\"" << Amanzi::AmanziInput::GlobalData::AMR_to_Amanzi_label_map[name]
			<< "\": (" << times.back() << ", " << vals[times.size()-1] << ")\n";
	      std::cout.precision(old_prec);
	    }
        }
    }
  val_old = val_new;
}

static IntVect
Index (const PointRegion& region,
       int                lev,
       const Amr*         amr)
{
    BL_ASSERT(amr != 0);
    BL_ASSERT(lev >= 0 && lev <= amr->finestLevel());

    IntVect iv;

    const Geometry& geom = amr->Geom(lev);

    for (int d=0; d<BL_SPACEDIM; ++d) {
        Real eps = 1.e-8*geom.CellSize(d);
        Real loc = std::max(geom.ProbLo(d)+eps, region.coor[d]);
        loc = std::min(loc, geom.ProbHi(d)-eps);

        iv[d] = floor((loc-geom.ProbLo(d))/geom.CellSize(d));
    }
    iv += geom.Domain().smallEnd();
    return iv;
}

Real
Observation::point_sample (Real time)
{
  const int finest_level = amrp->finestLevel();
  const Array<IntVect>& refRatio = amrp->refRatio();


  const Region* regPtr = &region;
  const PointRegion* ptreg = dynamic_cast<const PointRegion*>(regPtr);
  if (ptreg == 0) 
  {
      BoxLib::Abort("Point Sample observation requires a point region");
  }

  int proc_with_data = -1;
  Real value;
  Array<IntVect> idxs(finest_level+1);
  for (int lev = finest_level; lev >= 0 && proc_with_data<0; lev--)
  {
      // Compute IntVect at this level of cell containing this point
      idxs[lev] = Index(*ptreg,lev,amrp);

      // Decide if this processor owns a fab at this level containing this point
      //
      // FIXME: Do test on grids, derive when necessary
      //
      int nGrow = 0;
      const MultiFab* S_new = amrp->getLevel(lev).derive(field,time,nGrow);

      for (MFIter mfi(*S_new); mfi.isValid() && proc_with_data<0; ++mfi)
      {
          const Box& box = mfi.validbox();

          if (box.contains(idxs[lev]))
          {
              proc_with_data = ParallelDescriptor::MyProc();
              value = (*S_new)[mfi](idxs[lev],0);
          }
      }
      delete S_new;
      ParallelDescriptor::ReduceIntMax(proc_with_data);
  }

  if (proc_with_data<0) {
      if (ParallelDescriptor::IOProcessor()) {
          std::cout << region << std::endl;

          for (int lev = 0; lev <= finest_level; lev++)
          {
              std::cout << "   maps to " << idxs[lev] << " on AMR level " << lev
                        << " with grids: " << amrp->getLevel(lev).boxArray() << std::endl;
          }
      }
      BoxLib::Abort("Nobody claimed ownership of the observation pt");
  }
  ParallelDescriptor::Bcast(&value,1,proc_with_data);
  return value;
}

std::pair<Real,Real>
Observation::integral_and_volume (Real time)
{
  const int finest_level = amrp->finestLevel();
  const Array<IntVect>& refRatio = amrp->refRatio();

  Real int_inside = 0.0;
  Real vol_inside = 0;  
  
  Real vol_scale_lev = 1;

  for (int lev = 0; lev <= finest_level; lev++)
    {
      int nGrow = 0;
      const MultiFab* S = amrp->getLevel(lev).derive(field,time,nGrow);
      BoxArray baf;
      if (lev < finest_level)
        {
          BoxArray baf = amrp->boxArray(lev+1);
          baf.coarsen(refRatio[lev]);
        }

      FArrayBox fabVOL, fabINT;
      const Real* dx = amrp->Geom(lev).CellSize();
      Real vol = 1.;
      for (int i=0;i<BL_SPACEDIM;i++) vol *= dx[i];

      for (MFIter mfi(*S); mfi.isValid(); ++mfi)
        {
          const Box& cbox = mfi.validbox();
          fabVOL.resize(cbox,1);

          // Initialize to zero everywhere, then set to 1 inside region
          fabVOL.setVal(0);
          region.setVal(fabVOL,vol,0,dx,0);

          // Zero where covered with better data
          if (lev < finest_level)
            {
              std::vector< std::pair<int,Box> > isects = baf.intersections(cbox);
              
              for (int ii = 0, N = isects.size(); ii < N; ii++)
                {
                  fabVOL.setVal(0,isects[ii].second,0,fabVOL.nComp());
                }
            }

          // Now increment volume and integral inside
          vol_inside += fabVOL.sum(0);
          
          fabINT.resize(cbox,1);
          fabINT.copy((*S)[mfi],0,0,1);
          fabINT.mult(fabVOL);
          int_inside += fabINT.sum(0);
        }

      delete S;
    }
  ParallelDescriptor::ReduceRealSum(vol_inside);
  ParallelDescriptor::ReduceRealSum(int_inside);
  return std::pair<Real,Real>(int_inside,vol_inside);
}

Real
Observation::peak_sample (Real time)
{
  Real peak_val = - std::numeric_limits<Real>::max();
  const int finest_level = amrp->finestLevel();
  const Array<IntVect>& refRatio = amrp->refRatio();

  std::vector< std::pair<int,Box> > isects;
  for (int lev = 0; lev <= finest_level; lev++) {

    const Real* dx = amrp->Geom(lev).CellSize();
    int nGrow = 0;
    const MultiFab* S = amrp->getLevel(lev).derive(field,time,nGrow);

    FArrayBox mask;
    BoxArray cfba;
    if (lev < finest_level) {
      cfba = BoxArray(amrp->getLevel(lev).boxArray()).coarsen(refRatio[lev]);
    }

    for (MFIter mfi(*S); mfi.isValid(); ++mfi) {
      const FArrayBox& fab = (*S)[mfi];
      const Box& box = mfi.validbox();
      mask.resize(box,1);
      mask.setVal(0);
      region.setVal(mask,1,0,dx,0);

      // Mask out covered data
      if (lev < finest_level) {
	cfba.intersections(box,isects);
	for (int i=0, N=isects.size(); i<N; ++i) {
	  mask.setVal(0,isects[i].second,0,1);
	}
      }

      // Do check only over relevant cells
      for (IntVect iv=box.smallEnd(), End=box.bigEnd(); iv<=End; box.next(iv)) {
        if (mask(iv,0) != 0) {
	  peak_val = std::max(peak_val,fab(iv,0));
        }
      }
    }
    delete S;
  }
  ParallelDescriptor::ReduceRealMax(peak_val);
  return peak_val;
}

Real
Observation::average (Real time)
{
  std::pair<Real,Real> int_vol = integral_and_volume(time);
  return (int_vol.second == 0 ? 0 : int_vol.first / int_vol.second);
}

Real
Observation::volume_integral (Real time)
{
  return integral_and_volume(time).first;
}

Real
Observation::volume_time_integral (Real t_old, Real t_new)
{
  // Use trapezoid rule in time on volume integrals.
  std::pair<Real,Real> int_vol_0 = integral_and_volume(t_old);
  std::pair<Real,Real> int_vol_1 = integral_and_volume(t_new);

  return 0.5*(t_new-t_old)*(int_vol_0.first + int_vol_1.first);
}
