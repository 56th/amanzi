/*
  Geometry

  Copyright 2010-2012 held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Rao Garimella
           Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#ifndef AMANZI_GEOMETRY_POINT_HH_
#define AMANZI_GEOMETRY_POINT_HH_

#include <iostream>
#include <vector>
#include <cmath>

#include <Kokkos_Core.hpp>

#include "Kokkos_ArithTraits.hpp"

#include "dbc.hh"

namespace Amanzi {
namespace AmanziGeometry {

class Point {
 public:
  Point() {
    d = 0;
    xyz[0] = xyz[1] = xyz[2] = 0.0;
  }
  Point(const Point& p) {
    d = p.d;
    std::copy(p.xyz, p.xyz+d, xyz);
  }
  Point(volatile const Point& p) {
    d = p.d;
    std::copy(p.xyz, p.xyz+d, xyz);
  }

  Point(const int N) {
    d = N;
    xyz[0] = xyz[1] = xyz[2] = 0.0;    
  }
  Point(const double& x, const double& y) {
    d = 2;
    xyz[0] = x;
    xyz[1] = y;
  }
  Point(const double& x, const double& y, const double& z) {
    d = 3;
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;
  }
  ~Point() {};

  // main members
  void set(const double& val) {
    AMANZI_ASSERT(d > 0);
    for (int i = 0; i < d; i++) xyz[i] = val;
  }
  void set(const Point& p) {
    d = p.d;
    std::copy(p.xyz, p.xyz+d, xyz);
  }
  void set(const double *val) {
    AMANZI_ASSERT(val);
    AMANZI_ASSERT(d > 0);
    std::copy(val, val+d, xyz);
  }
  void set(const int N, const double *val) {
    AMANZI_ASSERT(val);
    d = N;
    std::copy(val,val+d,xyz);
  }
  void set(const double& x, const double& y) {
    d = 2;
    xyz[0] = x;
    xyz[1] = y;
  }
  void set(const double& x, const double& y, const double& z) {
    d = 3;
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;
  }

  int is_valid() { return (d == 2 || d == 3) ? 1 : 0; }

  // access members
  double& operator[] (const int i) { return xyz[i]; }
  const double& operator[] (const int i) const { return xyz[i]; }

  double x() const { return xyz[0]; }
  double y() const { return xyz[1]; }
  double z() const { return (d == 3) ? xyz[2] : 0.0; }

  int dim() const { return d; }


  // operators
  Point& operator=(const Point& p) {
    d = p.d;
    std::copy(p.xyz, p.xyz+d, xyz);
    return *this;
  }

  Point& operator=(const double& c) {
    xyz[0] = xyz[1] = xyz[2] = c;
    return *this;
  }

  volatile Point operator=(const Point& p) volatile {
    d = p.d;
    std::copy(p.xyz,p.xyz+d,xyz);
    return *this;
  }

  Point& operator*=(const double& c) {
    for (int i = 0; i < d; i++) xyz[i] *= c;
    return *this;
  }
  Point& operator/=(const double& c) {
    for (int i = 0; i < d; i++) xyz[i] /= c;
    return *this;
  }
  Point& operator+=(const double& c) {
    for (int i = 0; i < d; i++) xyz[i] += c;
    return *this;
  }
  Point& operator-=(const double& c) {
    for (int i = 0; i < d; i++) xyz[i] -= c;
    return *this;
  }

  Point& operator*=(const Point& p) {
    for (int i = 0; i < d; i++) xyz[i] *= p.xyz[i];
    return *this;
  }

  Point& operator+=(const Point& p) {
    for (int i = 0; i < d; i++) xyz[i] += p.xyz[i];
    return *this;
  }
  Point& operator/=(const Point& p) {
    for (int i = 0; i < d; i++) xyz[i] /= p.xyz[i];
    return *this;
  }
  Point& operator-=(const Point& p) {
    for (int i = 0; i < d; i++) xyz[i] -= p.xyz[i];
    return *this;
  }

  bool operator>(const Point&p) const {
    return xyz[0]>p.xyz[0] && xyz[1]>p.xyz[1] && xyz[2]>p.xyz[2];
  }

  bool operator<(const Point&p) const {
    return xyz[0]<p.xyz[0] && xyz[1]<p.xyz[1] && xyz[2]<p.xyz[2];
  }

  friend Point operator*(const double& r, const Point& p) {
    return (p.d == 2) ? Point(r*p[0], r*p[1]) : Point(r*p[0], r*p[1], r*p[2]);
  }

  friend Point operator*(const Point& p, const double& r) {
    return r*p;
  }

  friend Point operator+(const Point& p, const double& r) {
    return r+p;
  }

  friend Point operator+(const double& r, const Point& p) {
    return r+p;
  }

  friend double operator*(const Point& p, const Point& q) {
    double s = 0.0;
    for (int i = 0; i < p.d; i++ ) s += p[i]*q[i];
    return s;
  }

  friend Point operator/(const Point& p, const double& r) { return p * (1.0/r); }

  friend Point operator+(const Point& p, const Point& q) {
    return (p.d == 2) ? Point(p[0]+q[0], p[1]+q[1]) : Point(p[0]+q[0], p[1]+q[1], p[2]+q[2]);
  }
  friend Point operator-(const Point& p, const Point& q) {
    return (p.d == 2) ? Point(p[0]-q[0], p[1]-q[1]) : Point(p[0]-q[0], p[1]-q[1], p[2]-q[2]);
  }
  friend Point operator-(const Point& p) {
    return (p.d == 2) ? Point(-p[0], -p[1]) : Point(-p[0], -p[1], -p[2]);
  }
  friend Point operator/(const Point& p, const Point& q) {
    return (p.d == 2) ? Point(p[0]/q[0], p[1]/q[1]) : Point(p[0]/q[0], p[1]/q[1], p[2]/q[2]);
  }

  friend Point operator^(const Point& p, const Point& q) {
    Point pq(p.d);
    if (p.d == 2) {
      pq[0] = p[0] * q[1] - q[0] * p[1];
    } else if (p.d == 3) {
      pq[0] = p[1] * q[2] - p[2] * q[1];
      pq[1] = p[2] * q[0] - p[0] * q[2];
      pq[2] = p[0] * q[1] - p[1] * q[0];
    }
    return pq;
  }

  friend std::ostream& operator<<(std::ostream& os, const Point& p) {
    os << p.x() << " " << p.y();
    if (p.d == 3) os << " " << p.z();
    return os;
  }

  static Point this_type_is_missing_a_specialization(){}

 private:
  int d;
  double xyz[3];
};  // class Point


// Miscellaneous non-member functions
inline double L22(const Point& p) { return p*p; }
inline double norm(const Point& p) { return sqrt(p*p); }

inline bool operator==(const Point& p, const Point& q) {
  if (p.dim() != q.dim()) return false;
  for (int i = 0; i < p.dim(); ++i) 
    if (p[i] != q[i]) return false;
  return true;
}

inline bool operator!=(const Point& p, const Point& q) {
  return !(p == q);
}

}  // namespace AmanziGeometry
}  // namespace Amanzi

// Point specialization
template<>
struct Kokkos::ArithTraits<Amanzi::AmanziGeometry::Point> {
  using Point = Amanzi::AmanziGeometry::Point;
  typedef Point val_type;
  typedef Point mag_type;

  static inline Point nan(){
    return Point(std::nan(""),std::nan(""),std::nan(""));
  }
  static inline Point abs(const val_type& p){
    return Point(
      p[0]>=0?p[0]:-p[0],
      p[1]>=0?p[1]:-p[1],
      p[2]>=0?p[2]:-p[2]);
  }

  static inline Point one(){
    return Point(1,1,1);
  }

  static inline Point zero(){
    return Point(0,0,0);
  }

  static inline Point conj(const Point& p){
    return p;
  }

  static inline Point sqrt(const Point& p){
    return Point(std::sqrt(p[0]),std::sqrt(p[1]),std::sqrt(p[2]));
  }


};


#endif
