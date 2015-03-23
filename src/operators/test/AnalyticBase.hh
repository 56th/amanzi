/*
  This is the operators component of the Amanzi code.

  License: BSD
  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)

  Discrete source operator.
*/

#ifndef AMANZI_OPERATOR_ANALYTIC_BASE_HH_
#define AMANZI_OPERATOR_ANALYTIC_BASE_HH_

#include "Mesh.hh"

class AnalyticBase {
 public:
  AnalyticBase(Teuchos::RCP<const Amanzi::AmanziMesh::Mesh> mesh) : mesh_(mesh) {};
  ~AnalyticBase() {};

  virtual Amanzi::WhetStone::Tensor Tensor(const Amanzi::AmanziGeometry::Point& p, double t) = 0;
  virtual double pressure_exact(const Amanzi::AmanziGeometry::Point& p, double t) = 0;
  virtual Amanzi::AmanziGeometry::Point velocity_exact(const Amanzi::AmanziGeometry::Point& p, double t) = 0;
  virtual Amanzi::AmanziGeometry::Point gradient_exact(const Amanzi::AmanziGeometry::Point& p, double t) = 0;
  virtual double source_exact(const Amanzi::AmanziGeometry::Point& p, double t) = 0;

  void ComputeCellError(Epetra_MultiVector& p, double t, double& pnorm, double& l2_err, double& inf_err) {
    pnorm = 0.0;
    l2_err = 0.0;
    inf_err = 0.0;

    int ncells = mesh_->num_entities(Amanzi::AmanziMesh::CELL, Amanzi::AmanziMesh::OWNED);
    for (int c = 0; c < ncells; c++) {
      const Amanzi::AmanziGeometry::Point& xc = mesh_->cell_centroid(c);
      double tmp = pressure_exact(xc, t);
      double volume = mesh_->cell_volume(c);

      // std::cout << c << " " << tmp << " " << p[0][c] << std::endl;
      l2_err += std::pow(tmp - p[0][c], 2.0) * volume;
      inf_err = std::max(inf_err, fabs(tmp - p[0][c]));
      pnorm += std::pow(tmp, 2.0) * volume;
    }
#ifdef HAVE_MPI
    double tmp = pnorm;
    mesh_->get_comm()->SumAll(&tmp, &pnorm, 1);
    tmp = l2_err;
    mesh_->get_comm()->SumAll(&tmp, &l2_err, 1);
    tmp = inf_err;
    mesh_->get_comm()->MaxAll(&tmp, &inf_err, 1);
#endif
    pnorm = sqrt(pnorm);
    l2_err = sqrt(l2_err);
  }

  void ComputeFaceError(Epetra_MultiVector& u, double t, double& unorm, double& l2_err, double& inf_err) {
    unorm = 0.0;
    l2_err = 0.0;
    inf_err = 0.0;

    int nfaces = mesh_->num_entities(Amanzi::AmanziMesh::FACE, Amanzi::AmanziMesh::OWNED);
    for (int f = 0; f < nfaces; f++) {
      double area = mesh_->face_area(f);
      const Amanzi::AmanziGeometry::Point& normal = mesh_->face_normal(f);
      const Amanzi::AmanziGeometry::Point& xf = mesh_->face_centroid(f);
      const Amanzi::AmanziGeometry::Point& velocity = velocity_exact(xf, t);
      double tmp = velocity * normal;

      l2_err += std::pow((tmp - u[0][f]) / area, 2.0);
      inf_err = std::max(inf_err, fabs(tmp - u[0][f]) / area);
      unorm += std::pow(tmp / area, 2.0);
      // std::cout << f << " " << tmp << " " << u[0][f] << std::endl;
    }
#ifdef HAVE_MPI
    double tmp = unorm;
    mesh_->get_comm()->SumAll(&tmp, &unorm, 1);
    tmp = l2_err;
    mesh_->get_comm()->SumAll(&tmp, &l2_err, 1);
    tmp = inf_err;
    mesh_->get_comm()->MaxAll(&tmp, &inf_err, 1);
#endif
    unorm = sqrt(unorm);
    l2_err = sqrt(l2_err);
  }

  void ComputeNodeError(
      Epetra_MultiVector& p, double t,
      double& pnorm, double& l2_err, double& inf_err, double& hnorm, double& h1_err) {
    pnorm = 0.0;
    l2_err = 0.0;
    inf_err = 0.0;
    hnorm = 0.0;
    h1_err = 0.0;

    int d = mesh_->space_dimension();
    Amanzi::AmanziGeometry::Point xv(d);
    Amanzi::AmanziGeometry::Point grad(d);

    Amanzi::AmanziMesh::Entity_ID_List nodes;
    Amanzi::WhetStone::MFD3D_Diffusion mfd(mesh_);
    int ncells = mesh_->num_entities(Amanzi::AmanziMesh::CELL, Amanzi::AmanziMesh::OWNED);

    for (int c = 0; c < ncells; c++) {
      double volume = mesh_->cell_volume(c);

      mesh_->cell_get_nodes(c, &nodes);
      int nnodes = nodes.size();
      std::vector<double> cell_solution(nnodes);

      for (int k = 0; k < nnodes; k++) {
        int v = nodes[k];
        cell_solution[k] = p[0][v];

        mesh_->node_get_coordinates(v, &xv);
        double tmp = pressure_exact(xv, t);


        if (std::abs(tmp - p[0][v]) > .01) {
          Amanzi::AmanziGeometry::Point xv(2);
          mesh_->node_get_coordinates(v, &xv);
          // std::cout << v << " at " << xv << " error: " << tmp << " " << p[0][v] << std::endl;
        }
        l2_err += std::pow(tmp - p[0][v], 2.0) * volume / nnodes;
        inf_err = std::max(inf_err, tmp - p[0][v]);
        pnorm += std::pow(tmp, 2.0) * volume / nnodes;
      }

      const Amanzi::AmanziGeometry::Point& xc = mesh_->cell_centroid(c);
      const Amanzi::AmanziGeometry::Point& grad_exact = gradient_exact(xc, t);
      mfd.RecoverGradient_StiffnessMatrix(c, cell_solution, grad);

      h1_err += L22(grad - grad_exact) * volume;
      hnorm += L22(grad_exact) * volume;
    }
#ifdef HAVE_MPI
    double tmp = pnorm;
    mesh_->get_comm()->SumAll(&tmp, &pnorm, 1);
    tmp = l2_err;
    mesh_->get_comm()->SumAll(&tmp, &l2_err, 1);
    tmp = inf_err;
    mesh_->get_comm()->MaxAll(&tmp, &inf_err, 1);
    tmp = hnorm;
    mesh_->get_comm()->SumAll(&tmp, &hnorm, 1);
    tmp = h1_err;
    mesh_->get_comm()->SumAll(&tmp, &h1_err, 1);
#endif
    pnorm = sqrt(pnorm);
    l2_err = sqrt(l2_err);

    hnorm = sqrt(hnorm);
    h1_err = sqrt(h1_err);
  }

 protected:
  Teuchos::RCP<const Amanzi::AmanziMesh::Mesh> mesh_;
};

#endif

