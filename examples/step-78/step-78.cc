// The program starts with the usual include files, all of which you should have
// seen before by now:
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out_stack.h>
#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>

namespace BlackScholesSolver
{
  using namespace dealii;

  template <int dim>
  double coefficient(const Point<dim> &p)
  {
    return p.square();
  }

  template <int dim>
  Tensor<1, dim> SCoefficient(const Point<dim> &p)
  {
    return Tensor<1, dim>(p);
  }

  /*
  This Section creates a class for the known solution when testing using the MMS
  I am using $v(\tau,S) = -\tau^2 -S^2 + 6$ for my solution
  */
  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution(double maturity_time);
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
             const unsigned int component = 0) const override;

    Tensor<1, dim> time_deriv(const Point<dim> & p,
                              const unsigned int component = 0) const;

  private:
    double maturity_time;
  };


  template <int dim>
  double Solution<dim>::value(const Point<dim> & p,
                              const unsigned int component) const
  {
    return -std::pow(p(component), 2) - std::pow(this->get_time(), 2) + 6;
  }

  template <int dim>
  Tensor<1, dim> Solution<dim>::gradient(const Point<dim> & p,
                                         const unsigned int component) const
  {
    return Point<1>(-2 * p(component));
  }

  template <int dim>
  Tensor<1, dim> Solution<dim>::time_deriv(const Point<dim> & p,
                                           const unsigned int component) const
  {
    (void)component;
    Tensor<1, dim> return_value;
    for (int i = 0; dim; ++i)
      {
        return_value += -2 * (this->get_time()) * std::sin(p(i)) * Point<1>(1);
      }
    return return_value;
  }

  /*
  The next piece is the declaration of the main class of this program. This is
  very similar to the Step-26 tutorial, with some modifications. New matrices
  had to be added to calculate the A and B matrices, as well as the $V_{diff}$
  vector mentioned in the introduction.
  */
  template <int dim>
  class BlackScholes
  {
  public:
    BlackScholes();

    void run();

    /*
    Below are the parameters for the problem.
    's_max': The imposed upper bound on the spatial domain. This is the maximum
    allowed stock price.
    'maturity_time': The upper bound on the time domain. This is when the option
    expires.
    'sigma': The volatility of the stock price.
    'r': The risk free interest rate.
    'strike_price': The aggreed upon price that the buyer will have the option
    of purchasing  the stocks at the expiration time.
    */
    const double s_max;
    const double maturity_time;
    const double sigma;
    const double r;
    const double strike_price;

  private:
    void setup_system();
    void solve_time_step();
    void solve_time_step_diffusion();
    void refine_grid();
    void o_results();
    void process_solution(const unsigned int cycle);

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> a_matrix;
    SparseMatrix<double> b_matrix;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;


    double       time;
    double       time_step;
    unsigned int timestep_number;

    const double       theta;
    const unsigned int n_cycles;

    DataOutStack<dim>        data_out_stack;
    std::vector<std::string> solution_names;

    ConvergenceTable convergence_table;
  };

  /*
  In the following classes and functions, we implement the various pieces of
  data that define this problem. This includes the boundary values and the
  initial conditions that are used in this program and for which we need
  function objects. The initial conditions and boundary values as described in
  the introduction.
  */

  // Initial Condition Function
  template <int dim>
  class InitialConditions : public Function<dim>
  {
  public:
    InitialConditions(double s_price);
    double         s_price;
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double InitialConditions<dim>::value(const Point<dim> & p,
                                       const unsigned int component) const
  {
    (void)component;
    return -std::pow(p(component), 2) + 6;
    // Below is the original initial condition
    // return std::max(p(component) - s_price, 0.);
  }

  // Boundary Conditions Functions

  // Left Boundary Condition
  template <int dim>
  class LeftBoundaryValues : public Function<dim>
  {
  public:
    LeftBoundaryValues();
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double LeftBoundaryValues<dim>::value(const Point<dim> & p,
                                        const unsigned int component) const
  {
    (void)component;
    (void)p;
    return -std::pow(this->get_time(), 2) + 6;

    // Below is the original left boundary condition
    // return 0.0;
  }
  // Right Boundary Condition
  template <int dim>
  class RightBoundaryValues : public Function<dim>
  {
  public:
    RightBoundaryValues(double s_price, double r);
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    double s_price;
    double r;
  };


  template <int dim>
  double RightBoundaryValues<dim>::value(const Point<dim> & p,
                                         const unsigned int component) const
  {
    (void)component;
    return -std::pow(p(component), 2) - std::pow(this->get_time(), 2) + 6;
    // Below is the original right boundary condition
    // return (p(component) - _s_price) * exp((-_r) * (this->get_time()));
  }

  // Forcing Function (Right hand side function)
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide(double sigma, double r);
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    double sigma;
    double r;
  };

  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    // return 0;
    return 2 * (this->get_time()) - std::pow(sigma * p(component), 2) -
           2 * r * std::pow(p(component), 2) -
           r * (-std::pow(p(component), 2) - std::pow(this->get_time(), 2) + 6);

    // Below is the original right hand side
    // return 0.0
  }

  template <int dim>
  InitialConditions<dim>::InitialConditions(double s_price)
    : s_price(s_price)
  {}

  template <int dim>
  LeftBoundaryValues<dim>::LeftBoundaryValues()
  {}

  template <int dim>
  RightBoundaryValues<dim>::RightBoundaryValues(double s_price, double r)
    : s_price(s_price)
    , r(r)
  {}

  template <int dim>
  RightHandSide<dim>::RightHandSide(double sigma, double r)
    : sigma(sigma)
    , r(r)
  {}

  template <int dim>
  Solution<dim>::Solution(double maturity_time)
    : maturity_time(maturity_time)
  {}

  /*
  Now, we get to the implementation of the main class. This is the constructor,
  which sets the various parameters as described in the introduction.
  */
  template <int dim>
  BlackScholes<dim>::BlackScholes()
    : s_max(1.)
    , maturity_time(1.)
    , sigma(.2)
    , r(0.05)
    , strike_price(0.5)
    , fe(1)
    , dof_handler(triangulation)
    , time(0.0)
    , timestep_number(0)
    , theta(0.5)
    , n_cycles(3)
  {}

  /*
  The next function is the one that sets up the DoFHandler object, computes the
  constraints, and sets the linear algebra objects to their correct sizes. We
  also compute the mass matrix here by calling a function from the library. We
  will compute the other 3 matrices next, because these need to be computed 'by
  hand'.
  */
  template <int dim>
  void BlackScholes<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    // Initialize timestep here because I need maturity time already set.
    time_step = maturity_time / 5000.;

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    a_matrix.reinit(sparsity_pattern);
    b_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);

    /*
    Below is the code to create the Laplace matrix with non constant
    coefficients. This corresponds to the matrix D in the introduction. This
    non-constant coefficient is represented in the 'current_coefficient'
    variable.
    */
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    QGauss<dim>        quadrature_formula(fe.degree + 1);
    FEValues<dim>      fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0.;
        fe_values.reinit(cell);
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const double current_coefficient =
              coefficient(fe_values.quadrature_point(q_index));
            for (const unsigned int i : fe_values.dof_indices())
              {
                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (current_coefficient *              // a(x_q)
                     fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                     fe_values.JxW(q_index));           // dx
              }
          }
        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              laplace_matrix.add(local_dof_indices[i],
                                 local_dof_indices[j],
                                 cell_matrix(i, j));
          }
      }

    /*
    Below is the code to create the 'A' matrix as discussed in the introduction.
    The 'S' coefficient is represented in the 'current\_coefficient' variable.
    */
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0.;
        fe_values.reinit(cell);
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const Tensor<1, dim> current_coefficient =
              SCoefficient(fe_values.quadrature_point(q_index));
            for (const unsigned int i : fe_values.dof_indices())
              {
                for (const unsigned int j : fe_values.dof_indices())
                  {
                    cell_matrix(i, j) +=
                      (current_coefficient *               // a(x_q)
                       fe_values.shape_grad(i, q_index) *  // grad phi_i(x_q)
                       fe_values.shape_value(j, q_index) * // phi_j(x_q)
                       fe_values.JxW(q_index));            // dx
                  }
              }
          }
        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              a_matrix.add(local_dof_indices[i],
                           local_dof_indices[j],
                           cell_matrix(i, j));
          }
      }

    /*
    Below is the code to create the 'B' matrix as discussed in the introduction.
    The 'S' coefficient is represented in the 'current\_coefficient' variable.
    */
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0.;
        fe_values.reinit(cell);
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const Tensor<1, dim> current_coefficient =
              SCoefficient(fe_values.quadrature_point(q_index));
            for (const unsigned int i : fe_values.dof_indices())
              {
                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (current_coefficient *               // a(x_q)
                     fe_values.shape_value(i, q_index) * // phi_i(x_q)
                     fe_values.shape_grad(j, q_index) *  // grad phi_j(x_q)
                     fe_values.JxW(q_index));            // dx
              }
          }
        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              b_matrix.add(local_dof_indices[i],
                           local_dof_indices[j],
                           cell_matrix(i, j));
          }
      }


    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  /*
  The next function solves the timestep.
  */
  template <int dim>
  void BlackScholes<dim>::solve_time_step()
  {
    SolverControl                          solver_control(1000, 1e-12);
    SolverCG<Vector<double>>               cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);
    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
  }

  /*
  It is simply the function to build the solution together. For this, we
  create a new layer at each time, and then add the solution vector for that
  timestep. The function then stitches this together with the old solutions
  using 'build_patches'.
  */
  template <int dim>
  void BlackScholes<dim>::o_results()
  {
    data_out_stack.new_parameter_value(time, time_step);
    data_out_stack.attach_dof_handler(dof_handler);
    data_out_stack.add_data_vector(solution, solution_names);
    data_out_stack.build_patches(2);
    data_out_stack.finish_parameter_value();
  }

  /*
  This is somewhat unnecessary to have a function for the global refinement that
  we do. The reason for the function, is to allow for the possibility of an
  adaptive refinement later.
  */

  template <int dim>
  void BlackScholes<dim>::refine_grid()
  {
    triangulation.refine_global(1);
  }

  /*
  Now we get to the main driver of the program. This is where we do all the work
  of looping through the timesteps and calculating the solution vector each
  time. Here at the top, we set the initial refinement value and then create a
  mesh. Then we refine this mesh once. Next, we set up the data_out_stack object
  to store our solution. Finally we create a label for where we should start
  when rerunning the first timestep and interpolate the initial condition onto
  the mesh.
  */
  template <int dim>
  void BlackScholes<dim>::run()
  {
    const unsigned int initial_global_refinement = 0; // 5 8

    GridGenerator::hyper_cube(triangulation, 0.0, s_max, true);
    triangulation.refine_global(initial_global_refinement);



    /*
    This sets up the output data.
    */
    solution_names.emplace_back("u");
    data_out_stack.declare_data_vector(solution_names,
                                       DataOutStack<dim>::dof_vector);

    Vector<double> tmp;
    // Forcing terms
    Vector<double> forcing_terms;
    for (unsigned int cycle = 0; cycle < n_cycles; cycle++)
      {
        /*
        We need to reset the time and timestep for every cycle.
        */
        if (cycle != 0)
          {
            refine_grid();
            time            = 0.0;
            timestep_number = 0;
          }
        setup_system();
        tmp.reinit(solution.size());
        forcing_terms.reinit(solution.size());

        // Set the initial condition
        VectorTools::interpolate(dof_handler,
                                 InitialConditions<dim>(strike_price),
                                 old_solution);

        solution = old_solution;
        if (cycle == (n_cycles - 1))
          {
            o_results();
          }


        /*
        Next, we run the main loop which runs until we exceed the maturity time.
        We first need to solve the non-advection terms first, so we do that
        here. To do this, we first compute the right hand side of the equation,
        which is described in the introduction. Then we compute the left hand
        side of the equation that will need to be inverted to solve for
        $V_{diff}$. After all of this is done, we interpolate the boundary
        values and solve. This solution is stored into $V_{diff}$ to be used in
        the next part of the Lie Splitting.
        */
        while (time < maturity_time)
          {
            time += time_step;
            ++timestep_number;
            std::cout << "Time step " << timestep_number << " at t=" << time
                      << std::endl;

            mass_matrix.vmult(system_rhs, old_solution);

            laplace_matrix.vmult(tmp, old_solution);
            system_rhs.add((-1) * (1 - theta) * time_step * pow(sigma, 2) * 0.5,
                           tmp);
            mass_matrix.vmult(tmp, old_solution);

            system_rhs.add((-1) * (1 - theta) * time_step * r * 2, tmp);

            a_matrix.vmult(tmp, old_solution);
            system_rhs.add((-1) * time_step * r, tmp);

            b_matrix.vmult(tmp, old_solution);
            system_rhs.add((-1) * pow(sigma, 2) * time_step * 1, tmp);

            /*
            Calculate contribution of source terms.
            */
            RightHandSide<dim> rhs_function(sigma, r);
            rhs_function.set_time(time);
            VectorTools::create_right_hand_side(dof_handler,
                                                QGauss<dim>(fe.degree + 1),
                                                rhs_function,
                                                tmp);
            forcing_terms = tmp;
            forcing_terms *= time_step * theta;
            rhs_function.set_time(time - time_step);
            VectorTools::create_right_hand_side(dof_handler,
                                                QGauss<dim>(fe.degree + 1),
                                                rhs_function,
                                                tmp);
            forcing_terms.add(time_step * (1 - theta), tmp);
            system_rhs -= forcing_terms;

            // Now to compute the left side of the equation that needs to be
            // inverted.
            system_matrix.copy_from(mass_matrix);
            system_matrix.add((theta)*time_step * pow(sigma, 2) * 0.5,
                              laplace_matrix);
            system_matrix.add((time_step)*r * theta * (1 + 1), mass_matrix);

            constraints.condense(system_matrix, system_rhs);

            {
              RightBoundaryValues<dim> right_boundary_function(strike_price, r);
              LeftBoundaryValues<dim>  left_boundary_function;
              right_boundary_function.set_time(time);
              left_boundary_function.set_time(time);
              std::map<types::global_dof_index, double> boundary_values;
              VectorTools::interpolate_boundary_values(dof_handler,
                                                       0,
                                                       left_boundary_function,
                                                       boundary_values);
              VectorTools::interpolate_boundary_values(dof_handler,
                                                       1,
                                                       right_boundary_function,
                                                       boundary_values);
              MatrixTools::apply_boundary_values(boundary_values,
                                                 system_matrix,
                                                 solution,
                                                 system_rhs);
            }
            solve_time_step();

            /*
            We need to only output results on the last cycle
            */
            if (cycle == (n_cycles - 1))
              {
                o_results();
              }
            old_solution = solution;
          }

        process_solution(cycle);
      }

    const std::string filename = "solution.vtk";
    std::ofstream     output(filename);
    data_out_stack.write_vtk(output);



    // Convergence Table
    /*
    This next part is building the convergence table. First, we will create the
    headings and set up the cells properly. During this, we will also prescribe
    the precision of our results. Then we will write the calculated errors based
    on the L2, H1, and Linfinity norms to the console and to the error LaTex
    file.
    */
    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_precision("Linfty", 3);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("Linfty", true);
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
    convergence_table.set_tex_caption("H1", "@f$H^1@f$-error");
    convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
    std::string error_filename = "error";
    error_filename += "-global";
    error_filename += ".tex";
    std::ofstream error_table_file(error_filename);
    convergence_table.write_tex(error_table_file);

    /*
    Next, we will make the convergence table. We will again write this to the
    console and to the convergence LaTex file.
    */

    convergence_table.add_column_to_supercolumn("cells", "n cells");
    std::vector<std::string> new_order;
    new_order.emplace_back("n cells");
    new_order.emplace_back("H1");
    new_order.emplace_back("L2");
    convergence_table.set_column_order(new_order);
    convergence_table.evaluate_convergence_rates(
      "L2", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "H1", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "H1", ConvergenceTable::reduction_rate_log2);
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
    std::string conv_filename = "convergence";
    conv_filename += "-global";
    switch (fe.degree)
      {
        case 1:
          conv_filename += "-q1";
          break;
        case 2:
          conv_filename += "-q2";
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    conv_filename += ".tex";
    std::ofstream table_file(conv_filename);
    convergence_table.write_tex(table_file);
  }

  template <int dim>
  void BlackScholes<dim>::process_solution(const unsigned int cycle)
  {
    Solution<dim> sol(maturity_time);
    sol.set_time(time);
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      sol,
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 1),
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      sol,
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 1),
                                      VectorTools::H1_seminorm);
    const double H1_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_seminorm);
    const QTrapez<1>     q_trapez;
    const QIterated<dim> q_iterated(q_trapez, fe.degree * 2 + 1);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      sol,
                                      difference_per_cell,
                                      q_iterated,
                                      VectorTools::Linfty_norm);
    const double Linfty_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::Linfty_norm);
    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();
    std::cout << "Cycle " << cycle << ':' << std::endl
              << "   Number of active cells:       " << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: " << n_dofs << std::endl;
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("Linfty", Linfty_error);
  }
} // namespace BlackScholesSolver

int main()
{
  try
    {
      using namespace BlackScholesSolver;

      BlackScholes<1> black_scholes_solver;
      black_scholes_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
