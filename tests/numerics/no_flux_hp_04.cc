// ---------------------------------------------------------------------
//
// Copyright (C) 2013 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"



// a function that shows something useful on the surface of a sphere
template <int dim>
class RadialFunction : public Function<dim>
{
public:
  RadialFunction()
    : Function<dim>(dim)
  {}

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &v) const
  {
    Assert(v.size() == dim, ExcInternalError());

    switch (dim)
      {
        case 2:
          v(0) = p[0] + p[1];
          v(1) = p[1] - p[0];
          break;
        case 3:
          v(0) = p[0] + p[1];
          v(1) = p[1] - p[0];
          v(2) = p[2];
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
  }
};



template <int dim>
void
test(const Triangulation<dim> &tr, const hp::FECollection<dim> &fe)
{
  DoFHandler<dim> dof(tr);
  dof.distribute_dofs(fe);

  deallog << "FE=" << fe[0].get_name() << std::endl;

  std::set<types::boundary_id> boundary_ids;
  boundary_ids.insert(0);

  AffineConstraints<double> cm;
  VectorTools::compute_no_normal_flux_constraints(dof, 0, boundary_ids, cm);
  cm.close();

  DoFHandler<dim> dh(tr);
  dh.distribute_dofs(fe);

  Vector<double> v(dh.n_dofs());
  VectorTools::interpolate(dh, RadialFunction<dim>(), v);

  cm.distribute(v);

  // remove small entries
  for (unsigned int i = 0; i < v.size(); ++i)
    if (std::fabs(v(i)) < 1e-12)
      v(i) = 0;

  DataOut<dim, DoFHandler<dim>> data_out;
  data_out.attach_dof_handler(dh);

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);

  data_out.add_data_vector(v,
                           "x",
                           DataOut<dim, DoFHandler<dim>>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches(fe[0].degree);

  data_out.write_vtk(deallog.get_file_stream());
}



template <int dim>
void
test_hyper_sphere()
{
  Triangulation<dim> tr;
  GridGenerator::hyper_ball(tr);

  static const SphericalManifold<dim> boundary;
  tr.set_manifold(0, boundary);

  tr.refine_global(2);

  for (unsigned int degree = 1; degree < 6 - dim; ++degree)
    {
      hp::FECollection<dim> fe(
        FESystem<dim>(FE_Q<dim>(QIterated<1>(QTrapezoid<1>(), degree)), dim));
      test(tr, fe);
    }
}


int
main()
{
  initlog();
  deallog << std::setprecision(2);
  deallog << std::fixed;

  test_hyper_sphere<2>();
  test_hyper_sphere<3>();
}
