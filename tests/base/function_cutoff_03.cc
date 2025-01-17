// ---------------------------------------------------------------------
//
// Copyright (C) 2019 - 2020 by the deal.II authors
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

// Check that CutOffTensorProductFunction produces the right gradient in dim>1

#include <deal.II/base/function_lib.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include "../tests.h"

using namespace Functions;

template <int dim>
void
test()
{
  Point<dim> p;
  for (unsigned int i = 0; i < dim; ++i)
    p[i] = .5;


  CutOffFunctionTensorProduct<dim> fun(
    .5, p, 1, CutOffFunctionBase<dim>::no_component, true);


  fun.template set_base<CutOffFunctionC1>();

  deallog << "Center: " << fun.get_center() << std::endl
          << "Radius: " << fun.get_radius() << std::endl;

  // Check integration of the function, and of its gradient square
  QGauss<dim> quad(12);

  std::vector<Tensor<1, dim>> gradients(quad.size());
  std::vector<double>         values(quad.size());
  fun.value_list(quad.get_points(), values);
  fun.gradient_list(quad.get_points(), gradients);

  // Integral in 1d of grad square: 2.0*pi^2
  // Integral in 2d of grad square: 6.0*pi^2
  // Integral in 3d of grad square: 13.5*pi^2

  const double spi = numbers::PI * numbers::PI;

  const std::array<double, 3> exact{{2.0 * spi, 6.0 * spi, 13.5 * spi}};

  double integral      = 0.0;
  double grad_integral = 0.0;
  for (unsigned int i = 0; i < quad.size(); ++i)
    {
      integral += values[i] * quad.weight(i);
      grad_integral += gradients[i] * gradients[i] * quad.weight(i);
    }
  if (std::abs(integral - 1.0) > 1e-10)
    deallog << "Value NOT OK" << std::endl;
  else
    deallog << "Value OK" << std::endl;
  if (std::abs(grad_integral - exact[dim - 1]) > 1e-10)
    deallog << "Gradient NOT OK" << std::endl;
  else
    deallog << "Gradient OK" << std::endl;
}

int
main()
{
  initlog(1);

  test<1>();
  test<2>();
  test<3>();
}
