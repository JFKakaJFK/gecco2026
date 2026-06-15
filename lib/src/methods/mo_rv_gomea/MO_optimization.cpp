/**
 *
 * MO-RV-GOMEA
 *
 * If you use this software for any purpose, please cite the most recent publication:
 * A. Bouter, N.H. Luong, C. Witteveen, T. Alderliesten, P.A.N. Bosman. 2017.
 * The Multi-Objective Real-Valued Gene-pool Optimal Mixing Evolutionary Algorithm.
 * In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2017).
 * DOI: 10.1145/3071178.3071274
 *
 * Copyright (c) 1998-2017 Peter A.N. Bosman
 *
 * The software in this file is the proprietary information of
 * Peter A.N. Bosman.
 *
 * IN NO EVENT WILL THE AUTHOR OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 * The software in this file is the result of (ongoing) scientific research.
 * The following people have been actively involved in this research over
 * the years:
 * - Peter A.N. Bosman
 * - Dirk Thierens
 * - Jörn Grahl
 * - Anton Bouter
 *
 */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "MO_optimization.h"

namespace mo_rv_gomea_impl {
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Problems -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns the name of an installed problem.
 */
char* installedProblemName(int index) {
  switch (index) {
    case 0:
      return ((char*)"ZDT1");
    case 1:
      return ((char*)"ZDT2");
    case 2:
      return ((char*)"ZDT3");
    case 3:
      return ((char*)"ZDT4");
    case 4:
      return ((char*)"ZDT6");
    case 5:
      return ((char*)"BD1");
    case 6:
      return ((char*)"BD2 (scaled)");
    case 7:
      return ((char*)"genMED Convex 2D");
    case 8:
      return ((char*)"genMED Concave 2D");
    case 9:
      return ((char*)"Sum of ellipsoids");
    case 10:
      return ((char*)"User Defined Problem");
  }

  return (NULL);
}

/**
 * Returns the number of problems installed.
 */
int numberOfInstalledProblems(void) {
  static int result = -1;

  if (result == -1) {
    result = 0;
    while (installedProblemName(result) != NULL)
      result++;
  }

  return (result);
}

/**
 * Returns the number of objectives of an installed problem.
 */
int installedProblemNumberOfObjectives(int index) {
  switch (index) {
    case 0:
      return (2);
    case 1:
      return (2);
    case 2:
      return (2);
    case 3:
      return (2);
    case 4:
      return (2);
    case 5:
      return (2);
    case 6:
      return (2);
    case 7:
      return (2);
    case 8:
      return (2);
    case 9:
      return (2);
    case 10:
      assert(global_problem_ptr != NULL);
      return static_cast<int>(global_problem_ptr->num_objectives());
  }

  return (0);
}

/**
 * Returns the lower-range bound of an installed problem.
 */
double installedProblemLowerRangeBound(int index, int dimension) {
  switch (index) {
    case 0:
      return (ZDT1FunctionProblemLowerRangeBound(dimension));
    case 1:
      return (ZDT2FunctionProblemLowerRangeBound(dimension));
    case 2:
      return (ZDT3FunctionProblemLowerRangeBound(dimension));
    case 3:
      return (ZDT4FunctionProblemLowerRangeBound(dimension));
    case 4:
      return (ZDT6FunctionProblemLowerRangeBound(dimension));
    case 5:
      return (BD1FunctionProblemLowerRangeBound(dimension));
    case 6:
      return (BD2ScaledFunctionProblemLowerRangeBound(dimension));
    case 7:
      return (genMEDConvex2DFunctionProblemLowerRangeBound(dimension));
    case 8:
      return (genMEDConcave2DFunctionProblemLowerRangeBound(dimension));
    case 9:
      return (sumOfEllipsoidsFunctionProblemLowerRangeBound(dimension));
    case 10:
      assert(global_problem_ptr != NULL);
      return global_problem_ptr->continuous_lower_bounds()(dimension);
  }

  return (0.0);
}

/**
 * Returns the upper-range bound of an installed problem.
 */
double installedProblemUpperRangeBound(int index, int dimension) {
  switch (index) {
    case 0:
      return (ZDT1FunctionProblemUpperRangeBound(dimension));
    case 1:
      return (ZDT2FunctionProblemUpperRangeBound(dimension));
    case 2:
      return (ZDT3FunctionProblemUpperRangeBound(dimension));
    case 3:
      return (ZDT4FunctionProblemUpperRangeBound(dimension));
    case 4:
      return (ZDT6FunctionProblemUpperRangeBound(dimension));
    case 5:
      return (BD1FunctionProblemUpperRangeBound(dimension));
    case 6:
      return (BD2ScaledFunctionProblemUpperRangeBound(dimension));
    case 7:
      return (genMEDConvex2DFunctionProblemUpperRangeBound(dimension));
    case 8:
      return (genMEDConcave2DFunctionProblemUpperRangeBound(dimension));
    case 9:
      return (sumOfEllipsoidsFunctionProblemUpperRangeBound(dimension));
    case 10:
      assert(global_problem_ptr != NULL);
      return global_problem_ptr->continuous_upper_bounds()(dimension);
  }

  return (0.0);
}

/**
 * Returns whether a parameter is inside the range bound of
 * every problem.
 */
short isParameterInRangeBounds(double parameter, int dimension) {
  if (parameter < installedProblemLowerRangeBound(problem_index, dimension) ||
      parameter > installedProblemUpperRangeBound(problem_index, dimension) || isnan(parameter)) {
    return (0);
  }

  return (1);
}

/**
 * Initializes the parameter range bounds.
 */
void initializeParameterRangeBounds(void) {
  int i;

  lower_range_bounds = (double*)Malloc(number_of_parameters * sizeof(double));
  upper_range_bounds = (double*)Malloc(number_of_parameters * sizeof(double));
  lower_init_ranges = (double*)Malloc(number_of_parameters * sizeof(double));
  upper_init_ranges = (double*)Malloc(number_of_parameters * sizeof(double));

  for (i = 0; i < number_of_parameters; i++) {
    lower_range_bounds[i] = installedProblemLowerRangeBound(problem_index, i);
    upper_range_bounds[i] = installedProblemUpperRangeBound(problem_index, i);
  }

  for (i = 0; i < number_of_parameters; i++) {
    lower_init_ranges[i] = lower_user_range;
    if (lower_user_range < lower_range_bounds[i])
      lower_init_ranges[i] = lower_range_bounds[i];
    if (lower_user_range > upper_range_bounds[i])
      lower_init_ranges[i] = lower_range_bounds[i];

    upper_init_ranges[i] = upper_user_range;
    if (upper_user_range > upper_range_bounds[i])
      upper_init_ranges[i] = upper_range_bounds[i];
    if (upper_user_range < lower_range_bounds[i])
      upper_init_ranges[i] = upper_range_bounds[i];
  }
}

double repairParameter(double parameter, int dimension) {
  double result;

  result = parameter;
  result = fmax(result, installedProblemLowerRangeBound(problem_index, dimension));
  result = fmin(result, installedProblemUpperRangeBound(problem_index, dimension));

  return (result);
}

double distanceToRangeBounds(double* parameters) {
  int i;
  double d, sum;

  sum = 0.0;
  for (i = 0; i < number_of_parameters; i++) {
    if (!isParameterInRangeBounds(parameters[i], i)) {
      d = parameters[i] - repairParameter(parameters[i], i);
      sum += d * d;
    }
  }

  return (sqrt(sum));
}

void testEllipsoid(double* parameters, double* objective_value_result, double* constraint_value_result) {
  int i, j;
  double result;

  result = 0.0;
  for (i = 1; i < number_of_parameters; i++) {
    j = (i - 1) % block_size;
    result += pow(10.0, 6.0 * (((double)(j)) / ((double)(block_size - 1)))) * (parameters[i]) * (parameters[i]);
  }

  result /= sum_of_ellipsoids_normalization_factor;

  //
  result = 1.0 - parameters[0] + result;
  //
  //
  *objective_value_result = result;
  *constraint_value_result = 0;
}

/**
 * Compute the value of all objectives
 * and the sum of all constraint violations
 * function after rotating the parameter vector.
 * Both are returned using pointer variables.
 */
void installedProblemEvaluation(int index,
                                individual* ind,
                                int number_of_touched_parameters,
                                int* touched_parameters_indices,
                                double* parameters_before,
                                double* objective_values_before,
                                double constraint_value_before) {
  if (index == 10) {
    std::vector<size_t> idxs{0};
    goblin::InstanceBase& problem = *global_problem_ptr;
    goblin::ArchiveBase& archive = *global_archive_ptr;

    // MO-RV-GOMEA -> library solution & evaluation
    for (size_t i = 0; i < problem.num_continuous(); i++) {
      global_solution_set[0].continuous_values()(i) = ind->parameters[i];
    }

    number_of_full_evaluations++;
    if (touched_parameters_indices != NULL && !black_box_evaluations) {
      goblin::Subset touched;
      // create parent
      global_parent_set[0] = global_solution_set[0];
      // overwrite touched indices & collect subset
      for (size_t i = 0; i < number_of_touched_parameters; i++) {
        touched.continuous.push_back(touched_parameters_indices[i]);
        global_parent_set[0].continuous_values()(touched_parameters_indices[i]) = parameters_before[i];
      }
      // assign parent fitness
      auto& pq = global_parent_set[0].quality_as<goblin::MOQuality>();
      for (size_t i = 0; i < problem.num_objectives(); i++) {
        pq.objectives(i) = objective_values_before[i];
      }
      pq.constraint_value = constraint_value_before;

      // evaluate & update eval count
      std::vector<const goblin::Subset*> subsets{&touched};
      problem.evaluate_partial(*global_rng_ptr, global_solution_set, global_parent_set, subsets, idxs);

      number_of_evaluations += static_cast<double>(touched.size()) / static_cast<double>(problem.num_continuous());
    } else {
      number_of_evaluations++;
      problem.evaluate(*global_rng_ptr, global_solution_set, idxs);
    }

    // std::cout << problem.format_solution(global_solution_set[0]) << " @ "
    //           << problem.archive_fitness().format(global_solution_set[0].quality()) << std::endl;

    // termination status check
    archive.update(global_solution_set[0], true);
    if (problem.target_reached(archive)) {
      global_status = goblin::TerminationStatus::TargetReached;
      // hand control back to the wrapper, no matter where this is called within MO-RV-GOMEA
      // Note: the wonky manual short-circuiting instead of exceptions is to not mess with MO-RV-GOMEA state (would
      // crash/not free up memory)
      global_terminate_immediately = true;
    }

    // library quality -> MO-RV-GOMEA fitness
    auto& q = global_solution_set[0].quality_as<goblin::MOQuality>();
    for (size_t i = 0; i < problem.num_objectives(); i++) {
      ind->objective_values[i] = q.objectives(i);
    }
    ind->constraint_value = q.constraint_value;

    return;
  }

  int i, c, prev_block, cur_block, k, j, *block_indices;
  double *rotated_parameters, *touched_parameters, *rotated_parameters_before, *touched_parameters_before;

  touched_parameters = NULL;
  number_of_full_evaluations++;

  if (touched_parameters_indices != NULL && !black_box_evaluations) {
    touched_parameters = (double*)Malloc(number_of_touched_parameters * sizeof(double));
    for (i = 0; i < number_of_touched_parameters; i++)
      touched_parameters[i] = ind->parameters[touched_parameters_indices[i]];
    evaluateAdditionalFunctionsPartial(ind, number_of_touched_parameters, touched_parameters, parameters_before);
    number_of_evaluations += (double)number_of_touched_parameters / (double)number_of_parameters;
  } else {
    number_of_evaluations++;
  }

  for (i = 0; i < number_of_objectives; i++) {
    if (rotation_angle == 0.0) {
      if (touched_parameters_indices == NULL || black_box_evaluations)
        installedProblemEvaluationWithoutRotation(index, ind, ind->parameters, &(ind->objective_values[i]),
                                                  &(ind->constraint_value), number_of_parameters, NULL, NULL, NULL, 0,
                                                  i);
      else
        installedProblemEvaluationWithoutRotation(index, ind, touched_parameters, &(ind->objective_values[i]),
                                                  &(ind->constraint_value), number_of_touched_parameters,
                                                  touched_parameters_indices, parameters_before,
                                                  objective_values_before, constraint_value_before, i);
    } else {
      if (touched_parameters_indices == NULL || black_box_evaluations ||
          number_of_touched_parameters == number_of_parameters) {
        if (problem_index == 9)
          rotated_parameters = rotateParametersInRange(ind->parameters, 1, number_of_parameters - 1);
        else
          rotated_parameters = rotateAllParameters(ind->parameters);
        installedProblemEvaluationWithoutRotation(index, ind, rotated_parameters, &(ind->objective_values[i]),
                                                  &(ind->constraint_value), number_of_parameters, NULL, NULL, 0, 0, i);
        free(rotated_parameters);
      } else {
        if (problem_index == 9 && (learn_linkage_tree || use_univariate_FOS || static_linkage_tree)) {
          if (i == 0)
            sumOfEllipsoidsFunctionProblemEvaluationObjective0(ind, ind->parameters, &(ind->objective_values[0]),
                                                               &(ind->constraint_value));
          else if (i == 1) {
            if (touched_parameters_indices != NULL)
              free(touched_parameters);
            prev_block = -1;
            touched_parameters = (double*)Malloc(block_size * sizeof(double));
            touched_parameters_before = (double*)Malloc(block_size * sizeof(double));
            block_indices = (int*)Malloc(block_size * sizeof(int));

            for (k = 0; k < number_of_touched_parameters; k++) {
              if (touched_parameters_indices[k] == 0) {
                block_indices[0] = 0;
                touched_parameters[0] = ind->parameters[0];
                touched_parameters_before[0] = parameters_before[k];
                sumOfEllipsoidsFunctionPartialProblemEvaluationObjective1(
                    ind, touched_parameters, 1, block_indices, touched_parameters_before, ind->objective_values[1],
                    ind->constraint_value);
              } else {
                cur_block = (touched_parameters_indices[k] - 1) / block_size;

                if (cur_block != prev_block) {
                  for (j = 0; j < block_size; j++) {
                    block_indices[j] = 1 + cur_block * block_size + j;
                    touched_parameters[j] = ind->parameters[1 + cur_block * block_size + j];
                    touched_parameters_before[j] = ind->parameters[1 + cur_block * block_size + j];
                  }
                  c = 0;
                  while (k + c < number_of_touched_parameters &&
                         (touched_parameters_indices[k + c] - 1) / block_size == cur_block) {
                    touched_parameters_before[(touched_parameters_indices[k + c] - 1) % block_size] =
                        parameters_before[k + c];
                    c++;
                  }

                  rotated_parameters =
                      matrixVectorMultiplication(rotation_matrix, touched_parameters, block_size, block_size);
                  rotated_parameters_before =
                      matrixVectorMultiplication(rotation_matrix, touched_parameters_before, block_size, block_size);

                  sumOfEllipsoidsFunctionPartialProblemEvaluationObjective1(
                      ind, rotated_parameters, block_size, block_indices, rotated_parameters_before,
                      ind->objective_values[0], ind->constraint_value);

                  free(rotated_parameters_before);
                  free(rotated_parameters);
                }
                prev_block = cur_block;
              }
            }
            free(block_indices);
            free(touched_parameters_before);
          }
        } else {
          rotated_parameters = matrixVectorMultiplication(rotation_matrix, touched_parameters,
                                                          number_of_touched_parameters, number_of_touched_parameters);
          rotated_parameters_before = matrixVectorMultiplication(
              rotation_matrix, parameters_before, number_of_touched_parameters, number_of_touched_parameters);
          installedProblemEvaluationWithoutRotation(index, ind, rotated_parameters, &(ind->objective_values[i]),
                                                    &(ind->constraint_value), number_of_touched_parameters,
                                                    touched_parameters_indices, rotated_parameters_before,
                                                    objective_values_before, constraint_value_before, i);
          free(rotated_parameters_before);

          free(rotated_parameters);
        }
      }
    }
  }

  if (write_generational_statistics) {
    if ((int)(number_of_evaluations + 1) % 2000 == 0)
      evaluations_for_statistics_hit = 1;
  }

  if (touched_parameters_indices != NULL) {
    free(touched_parameters);
  }
}

/**
 * Returns the value of the single objective
 * and the sum of all constraint violations
 * without rotating the parameter vector.
 * Both are returned using pointer variables.
 */
void installedProblemEvaluationWithoutRotation(int index,
                                               individual* ind,
                                               double* parameters,
                                               double* objective_value_result,
                                               double* constraint_value_result,
                                               int number_of_touched_parameters,
                                               int* touched_parameters_indices,
                                               double* parameters_before,
                                               double* objective_values_before,
                                               double constraint_value_before,
                                               int objective_index) {
  if (black_box_evaluations || touched_parameters_indices == NULL) {
    switch (index) {
      case 0:
        ZDT1FunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result, objective_index);
        break;
      case 1:
        ZDT2FunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result, objective_index);
        break;
      case 2:
        ZDT3FunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result, objective_index);
        break;
      case 3:
        ZDT4FunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result, objective_index);
        break;
      case 4:
        ZDT6FunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result, objective_index);
        break;
      case 5:
        BD1FunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result, objective_index);
        break;
      case 6:
        BD2ScaledFunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result,
                                           objective_index);
        break;
      case 7:
        genMEDConvex2DFunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result,
                                                objective_index);
        break;
      case 8:
        genMEDConcave2DFunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result,
                                                 objective_index);
        break;
      case 9:
        sumOfEllipsoidsFunctionProblemEvaluation(ind, parameters, objective_value_result, constraint_value_result,
                                                 objective_index);
        break;

        // TODO
    }
  } else {
    switch (index) {
      case 0:
        ZDT1FunctionPartialProblemEvaluation(ind, parameters, objective_value_result, constraint_value_result,
                                             objective_index);
        break;
      case 1:
        ZDT2FunctionPartialProblemEvaluation(ind, parameters, objective_value_result, constraint_value_result,
                                             objective_index);
        break;
      case 2:
        ZDT3FunctionPartialProblemEvaluation(ind, parameters, objective_value_result, constraint_value_result,
                                             objective_index);
        break;
      case 3:
        ZDT4FunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result, objective_index);
        break;
      case 4:
        ZDT6FunctionPartialProblemEvaluation(ind, parameters, objective_value_result, constraint_value_result,
                                             objective_index);
        break;
      case 5:
        BD1FunctionPartialProblemEvaluation(ind, number_of_touched_parameters, touched_parameters_indices,
                                            parameters_before, objective_values_before, constraint_value_before,
                                            objective_index);
        break;
      case 6:
        BD2ScaledFunctionPartialProblemEvaluation(ind, number_of_touched_parameters, touched_parameters_indices,
                                                  parameters_before, objective_values_before, constraint_value_before,
                                                  objective_index);
        break;
      case 7:
        genMEDConvex2DFunctionPartialProblemEvaluation(ind, number_of_touched_parameters, touched_parameters_indices,
                                                       parameters_before, objective_values_before,
                                                       constraint_value_before, objective_index);
        break;
      case 8:
        genMEDConcave2DFunctionProblemEvaluation(parameters, objective_value_result, constraint_value_result,
                                                 objective_index);
        break;
        // case  9: sumOfEllipsoidsFunctionPartialProblemEvaluation( parameters, objective_value_result,
        // constraint_value_result, number_of_touched_parameters, touched_parameters_indices, parameters_before,
        // objective_values_before, constraint_value_before, objective_index ); break;

        // TODO
    }
  }
}

void evaluateAdditionalFunctionsFull(individual* ind) {
  int i;
  if (problem_index < 9) {
    ind->parameter_sum = 0;
    for (i = 0; i < number_of_parameters; i++)
      ind->parameter_sum += ind->parameters[i];
  }
}

void evaluateAdditionalFunctionsPartial(individual* ind,
                                        int number_of_touched_parameters,
                                        double* touched_parameters,
                                        double* parameters_before) {
  int i;
  if (problem_index != 9) {
    for (i = 0; i < number_of_touched_parameters; i++) {
      ind->parameter_sum += touched_parameters[i];
      ind->parameter_sum -= parameters_before[i];
    }
  }
}

void ZDT1FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index) {
  switch (objective_index) {
    case 0:
      ZDT1FunctionProblemEvaluationObjective0(parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      ZDT1FunctionProblemEvaluationObjective1(parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void ZDT1FunctionPartialProblemEvaluation(individual* ind,
                                          double* parameters,
                                          double* objective_value_result,
                                          double* constraint_value_result,
                                          int objective_index) {
  switch (objective_index) {
    case 0:
      ZDT1FunctionProblemEvaluationObjective0(ind->parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      ZDT1FunctionPartialProblemEvaluationObjective1(ind, parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void ZDT1FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  *objective_value_result = parameters[0];
  *constraint_value_result = 0;
}

void ZDT1FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  int i;
  double g, h, result;

  g = 0.0;
  for (i = 1; i < number_of_parameters; i++)
    g += parameters[i] / ((double)(number_of_parameters - 1.0));

  g = 1.0 + 9.0 * g;

  h = 1.0 - sqrt(parameters[0] / g);

  result = g * h;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

/*void ZDT1FunctionProblemEvaluationObjective1( double *parameters, double *objective_value_result, double
*constraint_value_result )
{
  int    i;
  double g, h, result, d;

  g = 0.0;
  for( i = 1; i < number_of_parameters; i++ )
    g += repairParameter(parameters[i],i)/((double) (number_of_parameters-1.0));

  g = 1.0+9.0*g;

  h = 1.0 - sqrt(repairParameter(parameters[0],0) / g);

  result = g*h;

  d = distanceToRangeBounds(parameters);
  result += 1e-6*d*d;

  *objective_value_result  = result;
  *constraint_value_result = 0;
}*/

void ZDT1FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                    double* parameters,
                                                    double* objective_value_result,
                                                    double* constraint_value_result) {
  double g, h, result;

  g = (ind->parameter_sum - ind->parameters[0]) / ((double)(number_of_parameters - 1.0));
  if (g < 0)
    g = 0;

  g = 1.0 + 9.0 * g;
  h = 1.0 - sqrt(ind->parameters[0] / g);

  result = g * h;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

double ZDT1FunctionProblemLowerRangeBound(int dimension) {
  return (0.0);
}

double ZDT1FunctionProblemUpperRangeBound(int dimension) {
  return (1.0);
}

void ZDT2FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index) {
  switch (objective_index) {
    case 0:
      ZDT2FunctionProblemEvaluationObjective0(parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      ZDT2FunctionProblemEvaluationObjective1(parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void ZDT2FunctionPartialProblemEvaluation(individual* ind,
                                          double* parameters,
                                          double* objective_value_result,
                                          double* constraint_value_result,
                                          int objective_index) {
  switch (objective_index) {
    case 0:
      ZDT2FunctionProblemEvaluationObjective0(ind->parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      ZDT2FunctionPartialProblemEvaluationObjective1(ind, parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void ZDT2FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  *objective_value_result = parameters[0];
  *constraint_value_result = 0;
}

void ZDT2FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  int i;
  double g, h, result;

  g = 0.0;
  for (i = 1; i < number_of_parameters; i++)
    g += parameters[i] / ((double)(number_of_parameters - 1.0));

  g = 1.0 + 9.0 * g;

  h = 1.0 - (parameters[0] / g) * (parameters[0] / g);

  result = g * h;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

void ZDT2FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                    double* parameters,
                                                    double* objective_value_result,
                                                    double* constraint_value_result) {
  double g, h, result;

  g = (ind->parameter_sum - ind->parameters[0]) / ((double)(number_of_parameters - 1.0));

  if (g < 0)
    g = 0;

  g = 1.0 + 9.0 * g;

  h = 1.0 - (ind->parameters[0] / g) * (ind->parameters[0] / g);

  result = g * h;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

double ZDT2FunctionProblemLowerRangeBound(int dimension) {
  return (0.0);
}

double ZDT2FunctionProblemUpperRangeBound(int dimension) {
  return (1.0);
}

void ZDT3FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index) {
  switch (objective_index) {
    case 0:
      ZDT3FunctionProblemEvaluationObjective0(parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      ZDT3FunctionProblemEvaluationObjective1(parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void ZDT3FunctionPartialProblemEvaluation(individual* ind,
                                          double* parameters,
                                          double* objective_value_result,
                                          double* constraint_value_result,
                                          int objective_index) {
  switch (objective_index) {
    case 0:
      ZDT3FunctionProblemEvaluationObjective0(ind->parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      ZDT3FunctionPartialProblemEvaluationObjective1(ind, parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void ZDT3FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  *objective_value_result = parameters[0];
  *constraint_value_result = 0;
}

void ZDT3FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  int i;
  double g, h, result;

  g = 0.0;
  for (i = 1; i < number_of_parameters; i++)
    g += parameters[i] / ((double)(number_of_parameters - 1.0));

  g = 1.0 + 9.0 * g;

  h = 1.0 - sqrt(parameters[0] / g) - (parameters[0] / g) * sin(10.0 * 3.1415926536 * parameters[0]);

  result = g * h;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

void ZDT3FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                    double* parameters,
                                                    double* objective_value_result,
                                                    double* constraint_value_result) {
  double g, h, result;

  g = (ind->parameter_sum - ind->parameters[0]) / ((double)(number_of_parameters - 1.0));

  if (g < 0)
    g = 0;

  g = 1.0 + 9.0 * g;

  h = 1.0 - sqrt(ind->parameters[0] / g) - (ind->parameters[0] / g) * sin(10.0 * 3.1415926536 * ind->parameters[0]);

  result = g * h;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

double ZDT3FunctionProblemLowerRangeBound(int dimension) {
  return (0.0);
}

double ZDT3FunctionProblemUpperRangeBound(int dimension) {
  return (1.0);
}

void ZDT4FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index) {
  switch (objective_index) {
    case 0:
      ZDT4FunctionProblemEvaluationObjective0(parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      ZDT4FunctionProblemEvaluationObjective1(parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void ZDT4FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  *objective_value_result = parameters[0];
  *constraint_value_result = 0;
}

void ZDT4FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  int i;
  double g, h, result;

  g = 0.0;
  for (i = 1; i < number_of_parameters; i++)
    g += parameters[i] * parameters[i] - 10.0 * cos(4.0 * 3.1415926536 * parameters[i]);

  g = 1.0 + 10.0 * (number_of_parameters - 1) + g;

  h = 1.0 - sqrt(parameters[0] / g);

  result = g * h;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

double ZDT4FunctionProblemLowerRangeBound(int dimension) {
  if (dimension == 0)
    return (0.0);

  return (-5.0);
}

double ZDT4FunctionProblemUpperRangeBound(int dimension) {
  if (dimension == 0)
    return (1.0);

  return (5.0);
}

void ZDT6FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index) {
  switch (objective_index) {
    case 0:
      ZDT6FunctionProblemEvaluationObjective0(parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      ZDT6FunctionProblemEvaluationObjective1(parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void ZDT6FunctionPartialProblemEvaluation(individual* ind,
                                          double* parameters,
                                          double* objective_value_result,
                                          double* constraint_value_result,
                                          int objective_index) {
  switch (objective_index) {
    case 0:
      ZDT6FunctionProblemEvaluationObjective0(ind->parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      ZDT6FunctionPartialProblemEvaluationObjective1(ind, parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void ZDT6FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  *objective_value_result = 1.0 - exp(-4.0 * parameters[0]) * pow(sin(6.0 * 3.1415926536 * parameters[0]), 6.0);
  *constraint_value_result = 0;
}

void ZDT6FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result) {
  int i;
  double g, h, f0, dummy, result;

  g = 0.0;
  for (i = 1; i < number_of_parameters; i++)
    g += parameters[i] / ((double)(number_of_parameters - 1));

  g = 1.0 + 9.0 * pow(g, 0.25);

  ZDT6FunctionProblemEvaluationObjective0(parameters, &f0, &dummy);

  h = 1.0 - (f0 / g) * (f0 / g);

  result = g * h;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

void ZDT6FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                    double* parameters,
                                                    double* objective_value_result,
                                                    double* constraint_value_result) {
  double g, h, f0, dummy, result;

  g = (ind->parameter_sum - ind->parameters[0]) / ((double)(number_of_parameters - 1.0));

  if (g < 0)
    g = 0;

  g = 1.0 + 9.0 * pow(g, 0.25);

  ZDT6FunctionProblemEvaluationObjective0(ind->parameters, &f0, &dummy);

  h = 1.0 - (f0 / g) * (f0 / g);

  result = g * h;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

double ZDT6FunctionProblemLowerRangeBound(int dimension) {
  return (0.0);
}

double ZDT6FunctionProblemUpperRangeBound(int dimension) {
  return (1.0);
}

void BD1FunctionProblemEvaluation(double* parameters,
                                  double* objective_value_result,
                                  double* constraint_value_result,
                                  int objective_index) {
  switch (objective_index) {
    case 0:
      BD1FunctionProblemEvaluationObjective0(parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      BD1FunctionProblemEvaluationObjective1(parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void BD1FunctionPartialProblemEvaluation(individual* ind,
                                         int number_of_touched_parameters,
                                         int* touched_parameters_indices,
                                         double* parameters_before,
                                         double* objective_values_before,
                                         double constraint_value_before,
                                         int objective_index) {
  switch (objective_index) {
    case 0:
      BD1FunctionProblemEvaluationObjective0(ind->parameters, &(ind->objective_values[0]), &(ind->constraint_value));
      break;
    case 1:
      BD1FunctionPartialProblemEvaluationObjective1(ind, number_of_touched_parameters, touched_parameters_indices,
                                                    parameters_before, objective_values_before,
                                                    constraint_value_before);
      break;
  }
}

void BD1FunctionProblemEvaluationObjective0(double* parameters,
                                            double* objective_value_result,
                                            double* constraint_value_result) {
  *objective_value_result = parameters[0];
  *constraint_value_result = 0;
}

void BD1FunctionProblemEvaluationObjective1(double* parameters,
                                            double* objective_value_result,
                                            double* constraint_value_result) {
  int i;
  double sum, result;

  sum = 0.0;
  for (i = 1; i < number_of_parameters - 1; i++)
    sum += 100.0 * (parameters[i + 1] - parameters[i] * parameters[i]) *
               (parameters[i + 1] - parameters[i] * parameters[i]) +
           (1.0 - parameters[i]) * (1.0 - parameters[i]);

  result = 1.0 - parameters[0] + sum;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

void BD1FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                   int number_of_touched_parameters,
                                                   int* touched_parameters_indices,
                                                   double* parameters_before,
                                                   double* objective_values_before,
                                                   double constraint_value_before) {
  int i, index;
  double sum, result, pi_before, pi_plusone_before, pi_minusone_before;

  sum = objective_values_before[1] - 1.0;
  if (touched_parameters_indices[0] == 0)
    sum += parameters_before[0];
  else
    sum += ind->parameters[0];

  if (sum < 0)
    sum = 0;

  for (i = 0; i < number_of_touched_parameters; i++) {
    index = touched_parameters_indices[i];
    if (index == 0)
      continue;
    if (index > 1) {
      pi_before = parameters_before[i];
      pi_minusone_before = ind->parameters[index - 1];
      if (i > 0 && touched_parameters_indices[i - 1] == index - 1) {
        pi_minusone_before = parameters_before[i - 1];
      } else {
        sum += 100.0 * (ind->parameters[index] - ind->parameters[index - 1] * ind->parameters[index - 1]) *
                   (ind->parameters[index] - ind->parameters[index - 1] * ind->parameters[index - 1]) +
               (1.0 - ind->parameters[index - 1]) * (1.0 - ind->parameters[index - 1]);
        sum -= 100.0 * (pi_before - pi_minusone_before * pi_minusone_before) *
                   (pi_before - pi_minusone_before * pi_minusone_before) +
               (1.0 - pi_minusone_before) * (1.0 - pi_minusone_before);
        if (sum < 0)
          sum = 0;
      }
    }

    if (index < number_of_parameters - 1) {
      pi_before = parameters_before[i];
      pi_plusone_before = ind->parameters[index + 1];
      if (i < number_of_touched_parameters - 1 && touched_parameters_indices[i + 1] == index + 1)
        pi_plusone_before = parameters_before[i + 1];
      sum += 100.0 * (ind->parameters[index + 1] - ind->parameters[index] * ind->parameters[index]) *
                 (ind->parameters[index + 1] - ind->parameters[index] * ind->parameters[index]) +
             (1.0 - ind->parameters[index]) * (1.0 - ind->parameters[index]);
      sum -= 100.0 * (pi_plusone_before - pi_before * pi_before) * (pi_plusone_before - pi_before * pi_before) +
             (1.0 - pi_before) * (1.0 - pi_before);

      if (sum < 0)
        sum = 0;
    }
  }

  result = 1.0 - ind->parameters[0] + sum;

  ind->objective_values[1] = result;
  ind->constraint_value = 0;
}

double BD1FunctionProblemLowerRangeBound(int dimension) {
  if (dimension == 0)
    return (0.0);

  return (-1e+308);
}

double BD1FunctionProblemUpperRangeBound(int dimension) {
  if (dimension == 0)
    return (1.0);

  return (1e+308);
}

void BD2ScaledFunctionProblemEvaluation(double* parameters,
                                        double* objective_value_result,
                                        double* constraint_value_result,
                                        int objective_index) {
  switch (objective_index) {
    case 0:
      BD2ScaledFunctionProblemEvaluationObjective0(parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      BD2ScaledFunctionProblemEvaluationObjective1(parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void BD2ScaledFunctionPartialProblemEvaluation(individual* ind,
                                               int number_of_touched_parameters,
                                               int* touched_parameters_indices,
                                               double* parameters_before,
                                               double* objective_values_before,
                                               double constraint_value_before,
                                               int objective_index) {
  switch (objective_index) {
    case 0:
      BD2ScaledFunctionPartialProblemEvaluationObjective0(ind, number_of_touched_parameters, touched_parameters_indices,
                                                          parameters_before, objective_values_before,
                                                          constraint_value_before);
      break;
    case 1:
      BD2ScaledFunctionPartialProblemEvaluationObjective1(ind, number_of_touched_parameters, touched_parameters_indices,
                                                          parameters_before, objective_values_before,
                                                          constraint_value_before);
      break;
  }
}

void BD2ScaledFunctionProblemEvaluationObjective0(double* parameters,
                                                  double* objective_value_result,
                                                  double* constraint_value_result) {
  int i;
  double sum, result;

  sum = 0.0;
  for (i = 0; i < number_of_parameters; i++)
    sum += parameters[i] * parameters[i];

  result = sum;
  result /= (double)number_of_parameters;

  *objective_value_result = result;
  *constraint_value_result = 0;
}

void BD2ScaledFunctionPartialProblemEvaluationObjective0(individual* ind,
                                                         int number_of_touched_parameters,
                                                         int* touched_parameters_indices,
                                                         double* parameters_before,
                                                         double* objective_values_before,
                                                         double constraint_value_before) {
  int i;
  double sum, result;

  sum = objective_values_before[0] * ((double)number_of_parameters);
  for (i = 0; i < number_of_touched_parameters; i++) {
    sum -= parameters_before[i] * parameters_before[i];
    sum += (ind->parameters[touched_parameters_indices[i]]) * (ind->parameters[touched_parameters_indices[i]]);
    if (sum < 0)
      sum = 0;
  }

  result = sum;
  result /= (double)number_of_parameters;

  ind->objective_values[0] = result;
  ind->constraint_value = 0;
}

void BD2ScaledFunctionProblemEvaluationObjective1(double* parameters,
                                                  double* objective_value_result,
                                                  double* constraint_value_result) {
  int i;
  double sum, result;

  sum = 0.0;
  for (i = 0; i < number_of_parameters - 1; i++)
    sum += 100.0 * (parameters[i + 1] - parameters[i] * parameters[i]) *
               (parameters[i + 1] - parameters[i] * parameters[i]) +
           (1.0 - parameters[i]) * (1.0 - parameters[i]);

  result = sum;
  result /= (double)(number_of_parameters - 1);

  *objective_value_result = result;
  *constraint_value_result = 0;
}

void BD2ScaledFunctionPartialProblemEvaluationObjective1(individual* ind,
                                                         int number_of_touched_parameters,
                                                         int* touched_parameters_indices,
                                                         double* parameters_before,
                                                         double* objective_values_before,
                                                         double constraint_value_before) {
  int i, index;
  double sum, result, pi_before, pi_plusone_before, pi_minusone_before;

  sum = objective_values_before[1] * ((double)(number_of_parameters - 1));

  for (i = 0; i < number_of_touched_parameters; i++) {
    index = touched_parameters_indices[i];
    if (index > 0) {
      pi_before = parameters_before[i];
      pi_minusone_before = ind->parameters[index - 1];
      if (i > 0 && touched_parameters_indices[i - 1] == index - 1) {
        pi_minusone_before = parameters_before[i - 1];
      } else {
        sum += 100.0 * (ind->parameters[index] - ind->parameters[index - 1] * ind->parameters[index - 1]) *
                   (ind->parameters[index] - ind->parameters[index - 1] * ind->parameters[index - 1]) +
               (1.0 - ind->parameters[index - 1]) * (1.0 - ind->parameters[index - 1]);
        sum -= 100.0 * (pi_before - pi_minusone_before * pi_minusone_before) *
                   (pi_before - pi_minusone_before * pi_minusone_before) +
               (1.0 - pi_minusone_before) * (1.0 - pi_minusone_before);
        if (sum < 0)
          sum = 0;
      }
    }

    if (index < number_of_parameters - 1) {
      pi_before = parameters_before[i];
      pi_plusone_before = ind->parameters[index + 1];
      if (i < number_of_touched_parameters - 1 && touched_parameters_indices[i + 1] == index + 1)
        pi_plusone_before = parameters_before[i + 1];
      sum += 100.0 * (ind->parameters[index + 1] - ind->parameters[index] * ind->parameters[index]) *
                 (ind->parameters[index + 1] - ind->parameters[index] * ind->parameters[index]) +
             (1.0 - ind->parameters[index]) * (1.0 - ind->parameters[index]);
      sum -= 100.0 * (pi_plusone_before - pi_before * pi_before) * (pi_plusone_before - pi_before * pi_before) +
             (1.0 - pi_before) * (1.0 - pi_before);
      if (sum < 0)
        sum = 0;
    }
  }
  result = sum;
  result /= (double)(number_of_parameters - 1);

  ind->objective_values[1] = result;
  ind->constraint_value = 0;
}

double BD2ScaledFunctionProblemLowerRangeBound(int dimension) {
  return (-1e+308);
}

double BD2ScaledFunctionProblemUpperRangeBound(int dimension) {
  return (1e+308);
}

double genMED_0FunctionEvaluation(double* parameters, double exponent) {
  int i;
  double result;
  static double* center = NULL;

  if (center == NULL) {
    center = (double*)Malloc(number_of_parameters * sizeof(double));
    for (i = 0; i < number_of_parameters; i++)
      center[i] = 0;
    center[0] = 1.0;
  }

  result = pow(distanceEuclidean(parameters, center, number_of_parameters) / sqrt(2.0), exponent);

  return (result);
}

double genMED_1FunctionEvaluation(double* parameters, double exponent) {
  int i;
  double result;
  static double* center = NULL;

  if (center == NULL) {
    center = (double*)Malloc(number_of_parameters * sizeof(double));
    for (i = 0; i < number_of_parameters; i++)
      center[i] = 0;
    center[1] = 1.0;
  }

  result = pow(distanceEuclidean(parameters, center, number_of_parameters) / sqrt(2.0), exponent);

  return (result);
}

void genMEDConvex2DFunctionProblemEvaluation(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result,
                                             int objective_index) {
  switch (objective_index) {
    case 0:
      genMEDConvex2DFunctionProblemEvaluationObjective0(parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      genMEDConvex2DFunctionProblemEvaluationObjective1(parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void genMEDConvex2DFunctionProblemEvaluationObjective0(double* parameters,
                                                       double* objective_value_result,
                                                       double* constraint_value_result) {
  *objective_value_result = genMED_0FunctionEvaluation(parameters, 2.0);
  *constraint_value_result = 0;
}

void genMEDConvex2DFunctionProblemEvaluationObjective1(double* parameters,
                                                       double* objective_value_result,
                                                       double* constraint_value_result) {
  *objective_value_result = genMED_1FunctionEvaluation(parameters, 2.0);
  *constraint_value_result = 0;
}

double genMEDConvex2DFunctionProblemLowerRangeBound(int dimension) {
  return (0.0);
}

double genMEDConvex2DFunctionProblemUpperRangeBound(int dimension) {
  return (1.0);
}

void genMEDConvex2DFunctionPartialProblemEvaluation(individual* ind,
                                                    int number_of_touched_parameters,
                                                    int* touched_parameters_indices,
                                                    double* parameters_before,
                                                    double* objective_values_before,
                                                    double constraint_value_before,
                                                    int objective_index) {
  int i, j;
  double result;

  result = objective_values_before[objective_index] * 2.0;
  for (i = 0; i < number_of_touched_parameters; i++) {
    j = touched_parameters_indices[i];
    if (touched_parameters_indices[i] == objective_index) {
      result -= (parameters_before[i] - 1.0) * (parameters_before[i] - 1.0);
      result += (ind->parameters[j] - 1.0) * (ind->parameters[j] - 1.0);
    } else {
      result -= parameters_before[i] * parameters_before[i];
      result += ind->parameters[j] * ind->parameters[j];
    }
  }

  ind->objective_values[objective_index] = 0.5 * result;
  ind->constraint_value = 0;
}

void genMEDConcave2DFunctionProblemEvaluation(double* parameters,
                                              double* objective_value_result,
                                              double* constraint_value_result,
                                              int objective_index) {
  switch (objective_index) {
    case 0:
      genMEDConcave2DFunctionProblemEvaluationObjective0(parameters, objective_value_result, constraint_value_result);
      break;
    case 1:
      genMEDConcave2DFunctionProblemEvaluationObjective1(parameters, objective_value_result, constraint_value_result);
      break;
  }
}

void genMEDConcave2DFunctionProblemEvaluationObjective0(double* parameters,
                                                        double* objective_value_result,
                                                        double* constraint_value_result) {
  *objective_value_result = genMED_0FunctionEvaluation(parameters, 0.5);
  *constraint_value_result = 0;
}

void genMEDConcave2DFunctionProblemEvaluationObjective1(double* parameters,
                                                        double* objective_value_result,
                                                        double* constraint_value_result) {
  *objective_value_result = genMED_1FunctionEvaluation(parameters, 0.5);
  *constraint_value_result = 0;
}

double genMEDConcave2DFunctionProblemLowerRangeBound(int dimension) {
  return (-1e+308);
}

double genMEDConcave2DFunctionProblemUpperRangeBound(int dimension) {
  return (1e+308);
}

void sumOfEllipsoidsFunctionProblemEvaluation(individual* ind,
                                              double* parameters,
                                              double* objective_value_result,
                                              double* constraint_value_result,
                                              int objective_index) {
  switch (objective_index) {
    case 0:
      sumOfEllipsoidsFunctionProblemEvaluationObjective0(ind, parameters, objective_value_result,
                                                         constraint_value_result);
      break;
    case 1:
      sumOfEllipsoidsFunctionProblemEvaluationObjective1(ind, parameters, objective_value_result,
                                                         constraint_value_result);
      break;
  }
}

void sumOfEllipsoidsFunctionProblemEvaluationObjective0(individual* ind,
                                                        double* parameters,
                                                        double* objective_value_result,
                                                        double* constraint_value_result) {
  ind->objective_values[0] = parameters[0];
  ind->constraint_value = 0;
}

void sumOfEllipsoidsFunctionProblemEvaluationObjective1(individual* ind,
                                                        double* parameters,
                                                        double* objective_value_result,
                                                        double* constraint_value_result) {
  int i, j;
  double result;

  result = 0.0;
  for (i = 1; i < number_of_parameters; i++) {
    j = (i - 1) % block_size;
    result += pow(10.0, 6.0 * (((double)(j)) / ((double)(block_size - 1)))) * (parameters[i]) * (parameters[i]);
  }

  //
  ind->parameter_sum = result;
  //

  result /= sum_of_ellipsoids_normalization_factor;
  result = 1.0 - parameters[0] + result;

  ind->objective_values[1] = result;
  ind->constraint_value = 0;
}

void sumOfEllipsoidsFunctionPartialProblemEvaluationObjective1(individual* ind,
                                                               double* parameters,
                                                               int number_of_touched_parameters,
                                                               int* touched_parameters_indices,
                                                               double* parameters_before,
                                                               double objective_value_before,
                                                               double constraint_value_before) {
  int i, j;
  double result;

  result = ind->parameter_sum;
  if (number_of_touched_parameters > 1) {
    for (i = 0; i < number_of_touched_parameters; i++) {
      j = (touched_parameters_indices[i] - 1) % block_size;
      result -=
          (pow(10.0, 6.0 * (((double)(j)) / ((double)(block_size - 1)))) * parameters_before[i] * parameters_before[i]);
      result += (pow(10.0, 6.0 * (((double)(j)) / ((double)(block_size - 1)))) * parameters[i] * parameters[i]);

      if (result < 0)
        result = 0;
    }

    ind->parameter_sum = result;
  }

  result /= sum_of_ellipsoids_normalization_factor;

  result = 1.0 - ind->parameters[0] + result;

  ind->objective_values[1] = result;
  ind->constraint_value = 0;
}

double sumOfEllipsoidsFunctionProblemLowerRangeBound(int dimension) {
  if (dimension == 0)
    return (0.0);
  return (-1e+308);
}

double sumOfEllipsoidsFunctionProblemUpperRangeBound(int dimension) {
  if (dimension == 0)
    return (1.0);

  return (1e+308);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Ranking -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns 1 if x constraint-Pareto-dominates y, 0 otherwise.
 * x is not better than y unless:
 * - x and y are both infeasible and x has a smaller sum of constraint violations, or
 * - x is feasible and y is not, or
 * - x and y are both feasible and x Pareto dominates y
 */
short constraintParetoDominates(double* objective_values_x,
                                double constraint_value_x,
                                double* objective_values_y,
                                double constraint_value_y) {
  short result;

  result = 0;

  if (constraint_value_x > 0) /* x is infeasible */
  {
    if (constraint_value_y > 0) /* Both are infeasible */
    {
      if (constraint_value_x < constraint_value_y)
        result = 1;
    }
  } else /* x is feasible */
  {
    if (constraint_value_y > 0) /* x is feasible and y is not */
      result = 1;
    else /* Both are feasible */
      result = paretoDominates(objective_values_x, objective_values_y);
  }

  return (result);
}

/**
 * Returns 1 if x Pareto-dominates y, 0 otherwise.
 */
short paretoDominates(double* objective_values_x, double* objective_values_y) {
  short strict;
  int i, result;

  result = 1;
  strict = 0;
  for (i = 0; i < number_of_objectives; i++) {
    if (fabs(objective_values_x[i] - objective_values_y[i]) >= 0.00001) {
      if (objective_values_x[i] > objective_values_y[i]) {
        result = 0;
        break;
      }
      if (objective_values_x[i] < objective_values_y[i])
        strict = 1;
    }
  }
  if (strict == 0 && result == 1)
    result = 0;

  return (result);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Initialization -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void initializeProblem(void) {
  switch (problem_index) {
    default:
      break;
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Metrics -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns whether the D_{Pf->S} metric can be computed.
 */
short haveDPFSMetric(void) {
  int default_front_size;

  getDefaultFront(&default_front_size);
  if (default_front_size > 0)
    return (1);

  return (0);
}

double** readDefaultFront(char* filename, int* default_front_size) {
  int i;
  FILE* f;
  static double** result = NULL;

  *default_front_size = 5000;
  f = fopen(filename, "r");

  if (f == NULL) {
    printf("No defaultFront file found.\n");
    fclose(f);
    *default_front_size = 0;
    return (NULL);
  }

  if (result == NULL) {
    result = (double**)Malloc((*default_front_size) * sizeof(double*));
    for (i = 0; i < (*default_front_size); i++) {
      result[i] = (double*)Malloc(2 * sizeof(double));
      fscanf(f, "%lf %lf\n", &result[i][0], &result[i][1]);
    }
  }

  fclose(f);

  return result;
}

/**
 * Returns the default front(NULL if there is none).
 * The number of solutions in the default
 * front is returned in the pointer variable.
 */
double** getDefaultFront(int* default_front_size) {
  *default_front_size = 0;

  switch (problem_index) {
    case 0:
      return (getDefaultFrontZDT1(default_front_size));
    case 1:
      return (getDefaultFrontZDT2(default_front_size));
    case 2:
      return (getDefaultFrontZDT3(default_front_size));
    case 3:
      return (getDefaultFrontZDT4(default_front_size));
    case 4:
      return (getDefaultFrontZDT6(default_front_size));
    case 5:
      return (getDefaultFrontBD1(default_front_size));
    case 6:
      return (getDefaultFrontBD2Scaled(default_front_size));
    case 7:
      return (getDefaultFrontGenMEDConvex(default_front_size));
    case 8:
      return (getDefaultFrontGenMEDConcave(default_front_size));
    case 9:
      return (getDefaultFrontBD1(default_front_size));
  }

  return (NULL);
}

double** getDefaultFrontZDT1(int* default_front_size) {
  char filename[100];
  sprintf(filename, "../defaultFronts/ZDT1.txt");
  return (readDefaultFront(filename, default_front_size));
}

double** getDefaultFrontZDT2(int* default_front_size) {
  char filename[100];
  sprintf(filename, "../defaultFronts/ZDT2.txt");
  return (readDefaultFront(filename, default_front_size));
}

double** getDefaultFrontZDT3(int* default_front_size) {
  char filename[100];
  sprintf(filename, "../defaultFronts/ZDT3.txt");
  return (readDefaultFront(filename, default_front_size));
}

double** getDefaultFrontZDT4(int* default_front_size) {
  char filename[100];
  sprintf(filename, "../defaultFronts/ZDT4.txt");
  return (readDefaultFront(filename, default_front_size));
}

double** getDefaultFrontZDT6(int* default_front_size) {
  char filename[100];
  sprintf(filename, "../defaultFronts/ZDT6.txt");
  return (readDefaultFront(filename, default_front_size));
}

double** getDefaultFrontBD1(int* default_front_size) {
  char filename[100];
  sprintf(filename, "../defaultFronts/BD1.txt");
  return (readDefaultFront(filename, default_front_size));
}

double** getDefaultFrontBD2(int* default_front_size) {
  char filename[100];
  sprintf(filename, "../defaultFronts/BD2.txt");
  return (readDefaultFront(filename, default_front_size));
}

double** getDefaultFrontBD2Scaled(int* default_front_size) {
  int i;
  double** default_front_BD2Scaled;
  static double** result = NULL;

  default_front_BD2Scaled = getDefaultFrontBD2(default_front_size);

  if (result == NULL) {
    result = (double**)Malloc((*default_front_size) * sizeof(double*));
    for (i = 0; i < (*default_front_size); i++)
      result[i] = (double*)Malloc(2 * sizeof(double));

    for (i = 0; i < (*default_front_size); i++) {
      result[i][0] = default_front_BD2Scaled[i][0] / 10.0;
      result[i][1] = default_front_BD2Scaled[i][1] / 9.0;
    }
    if (default_front_BD2Scaled != NULL) {
      for (i = 0; i < (*default_front_size); i++)
        free(default_front_BD2Scaled[i]);
      free(default_front_BD2Scaled);
    }
  }

  return (result);
}

double** getDefaultFrontGenMEDConvex(int* default_front_size) {
  int i;
  static double **result = NULL, t;

  *default_front_size = 5000;

  if (result == NULL) {
    result = (double**)Malloc((*default_front_size) * sizeof(double*));
    for (i = 0; i < (*default_front_size); i++)
      result[i] = (double*)Malloc(2 * sizeof(double));

    for (i = 0; i < (*default_front_size); i++) {
      t = ((double)i) / 4999.0;
      result[i][0] = pow(t, 2.0);
      result[i][1] = pow(1.0 - t, 2.0);
    }
  }

  return (result);
}

double** getDefaultFrontGenMEDConcave(int* default_front_size) {
  int i;
  static double **result = NULL, t;

  *default_front_size = 5000;

  if (result == NULL) {
    result = (double**)Malloc((*default_front_size) * sizeof(double*));
    for (i = 0; i < (*default_front_size); i++)
      result[i] = (double*)Malloc(2 * sizeof(double));

    for (i = 0; i < (*default_front_size); i++) {
      t = ((double)i) / 4999.0;
      result[i][0] = pow(t, 0.5);
      result[i][1] = pow(1.0 - t, 0.5);
    }
  }

  return (result);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*=-=-=-=-=-=-=-=-=-=-=-= Section Elitist Archive -==-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Adapts the objective box discretization. If the numbre
 * of solutions in the elitist archive is too high or too low
 * compared to the population size, the objective box
 * discretization is adjusted accordingly. In doing so, the
 * entire elitist archive is first emptied and then refilled.
 */
void adaptObjectiveDiscretization(void) {
  int i, j, k, na, nb, nc, elitist_archive_size_target_lower_bound, elitist_archive_size_target_upper_bound;
  double low, high, *elitist_archive_objective_ranges, elitist_archive_copy_size;
  individual** elitist_archive_copy;

  // printf("===========================================\n");
  // printf("Generation     : %d\n",number_of_generations);
  // printf("No improv. str.: %d\n",no_improvement_stretch);
  // printf("#Elitist target: %d\n",elitist_archive_size_target);
  // printf("#Elitist before: %d\n",elitist_archive_size);
  // printf("In effect      : %s\n",objective_discretization_in_effect?"true":"false");
  // printf("OBD before     : %e %e\n",objective_discretization[0],objective_discretization[1]);

  elitist_archive_size_target_lower_bound = (int)(0.75 * elitist_archive_size_target);
  elitist_archive_size_target_upper_bound = (int)(1.25 * elitist_archive_size_target);

  if (objective_discretization_in_effect && (elitist_archive_size < elitist_archive_size_target_lower_bound))
    objective_discretization_in_effect = 0;

  if (elitist_archive_size > elitist_archive_size_target_upper_bound) {
    objective_discretization_in_effect = 1;

    elitist_archive_objective_ranges = (double*)Malloc(number_of_objectives * sizeof(double));
    for (j = 0; j < number_of_objectives; j++) {
      low = elitist_archive[0]->objective_values[j];
      high = elitist_archive[0]->objective_values[j];

      for (i = 0; i < elitist_archive_size; i++) {
        if (elitist_archive[i]->objective_values[j] < low)
          low = elitist_archive[i]->objective_values[j];
        if (elitist_archive[i]->objective_values[j] > high)
          high = elitist_archive[i]->objective_values[j];
      }

      elitist_archive_objective_ranges[j] = high - low;
    }

    na = 1;
    nb = (int)pow(2.0, 25.0);
    for (k = 0; k < 25; k++) {
      elitist_archive_copy_size = elitist_archive_size;
      elitist_archive_copy = (individual**)Malloc(elitist_archive_copy_size * sizeof(individual*));
      for (i = 0; i < elitist_archive_copy_size; i++)
        elitist_archive_copy[i] = initializeIndividual();
      for (i = 0; i < elitist_archive_copy_size; i++) {
        copyIndividual(elitist_archive[i], elitist_archive_copy[i]);
      }

      nc = (na + nb) / 2;
      for (i = 0; i < number_of_objectives; i++)
        objective_discretization[i] = elitist_archive_objective_ranges[i] / ((double)nc);

      /* Restore the original elitist archive after the first cycle in this loop */
      if (k > 0) {
        elitist_archive_size = 0;
        for (i = 0; i < elitist_archive_copy_size; i++)
          addToElitistArchive(elitist_archive_copy[i], i);
      }

      /* Clear the elitist archive */
      elitist_archive_size = 0;

      /* Rebuild the elitist archive */
      for (i = 0; i < elitist_archive_copy_size; i++)
        updateElitistArchive(elitist_archive_copy[i]);

      if (elitist_archive_size <= elitist_archive_size_target_lower_bound)
        na = nc;
      else
        nb = nc;

      /* Copy the entire elitist archive */
      if (elitist_archive_copy != NULL) {
        for (i = 0; i < elitist_archive_copy_size; i++)
          ezilaitiniIndividual(elitist_archive_copy[i]);
        free(elitist_archive_copy);
      }
    }

    free(elitist_archive_objective_ranges);
  }
  // printf("In effect      : %s\n",objective_discretization_in_effect?"true":"false");
  // printf("OBD after      : %e %e\n",objective_discretization[0],objective_discretization[1]);
  // printf("#Elitist after : %d\n",elitist_archive_size);
  // printf("===========================================\n");
}

/**
 * Returns 1 if two solutions share the same objective box, 0 otherwise.
 */
short sameObjectiveBox(double* objective_values_a, double* objective_values_b) {
  int i;

  if (!objective_discretization_in_effect) {
    /* If the solutions are identical, they are still in the (infinitely small) same objective box. */
    for (i = 0; i < number_of_objectives; i++) {
      if (objective_values_a[i] != objective_values_b[i])
        return (0);
    }

    return (1);
  }

  for (i = 0; i < number_of_objectives; i++) {
    if (((int)(objective_values_a[i] / objective_discretization[i])) !=
        ((int)(objective_values_b[i] / objective_discretization[i]))) {
      return (0);
    }
  }

  return (1);
}

/**
 * Updates the elitist archive by offering a new solution
 * to possibly be added to the archive. If there are no
 * solutions in the archive yet, the solution is added.
 * Otherwise, the number of times the solution is
 * dominated is computed. Solution A is always dominated
 * by solution B that is in the same domination-box if
 * B dominates A or A and B do not dominate each other.
 * If the number of times a solution is dominated, is 0,
 * the solution is added to the archive and all solutions
 * dominated by the new solution, are purged from the archive.
 */
void updateElitistArchive(individual* ind) {
  short is_dominated_itself, is_extreme_compared_to_archive, all_to_be_removed;
  int i, j, *indices_dominated, number_of_solutions_dominated, insert_index;

  is_extreme_compared_to_archive = 0;
  all_to_be_removed = 1;
  insert_index = elitist_archive_size;
  if (ind->constraint_value == 0) {
    if (elitist_archive_size == 0) {
      is_extreme_compared_to_archive = 1;
    } else {
      for (j = 0; j < number_of_objectives; j++) {
        if (ind->objective_values[j] < best_objective_values_in_elitist_archive[j]) {
          is_extreme_compared_to_archive = 1;
          break;
        }
      }
    }
  }

  if (elitist_archive_size == 0)
    addToElitistArchive(ind, insert_index);
  else {
    indices_dominated = (int*)Malloc(elitist_archive_size * sizeof(int));
    number_of_solutions_dominated = 0;
    is_dominated_itself = 0;
    double* bla = (double*)Malloc(number_of_objectives * sizeof(double));
    bla[0] = 0.5;
    bla[1] = 0.5;
    for (i = 0; i < elitist_archive_size; i++) {
      if (elitist_archive_indices_inactive[i]) {
        if (i < insert_index)
          insert_index = i;
        continue;
      }
      all_to_be_removed = 0;
      if (constraintParetoDominates(elitist_archive[i]->objective_values, elitist_archive[i]->constraint_value,
                                    ind->objective_values, ind->constraint_value))
        is_dominated_itself = 1;
      else {
        if (!constraintParetoDominates(ind->objective_values, ind->constraint_value,
                                       elitist_archive[i]->objective_values, elitist_archive[i]->constraint_value)) {
          if (sameObjectiveBox(elitist_archive[i]->objective_values, ind->objective_values) &&
              (!is_extreme_compared_to_archive))
            is_dominated_itself = 1;
        }
      }

      if (is_dominated_itself)
        break;
    }
    free(bla);

    if (all_to_be_removed)
      addToElitistArchive(ind, insert_index);
    else if (!is_dominated_itself) {
      for (i = 0; i < elitist_archive_size; i++) {
        if (elitist_archive_indices_inactive[i])
          continue;
        if (constraintParetoDominates(ind->objective_values, ind->constraint_value,
                                      elitist_archive[i]->objective_values, elitist_archive[i]->constraint_value) ||
            sameObjectiveBox(elitist_archive[i]->objective_values, ind->objective_values)) {
          indices_dominated[number_of_solutions_dominated] = i;
          elitist_archive_indices_inactive[i] = 1;
          number_of_solutions_dominated++;
        }
      }

      if (number_of_solutions_dominated > 0) {
        if (ind->constraint_value == 0) {
          for (i = 0; i < number_of_solutions_dominated; i++) {
            for (j = 0; j < number_of_objectives; j++) {
              if (elitist_archive[indices_dominated[i]]->objective_values[j] ==
                  best_objective_values_in_elitist_archive[j]) {
                best_objective_values_in_elitist_archive[j] = ind->objective_values[j];
              }
            }
          }
        }
        removeFromElitistArchive(indices_dominated, number_of_solutions_dominated);
      }

      addToElitistArchive(ind, insert_index);
    }

    free(indices_dominated);
  }
}

void removeFromElitistArchive(int* indices, int number_of_indices) {
  int i;

  for (i = 0; i < number_of_indices; i++)
    elitist_archive_indices_inactive[indices[i]] = 1;
}

/**
 * Adds a solution to the elitist archive.
 */
void addToElitistArchive(individual* ind, int insert_index) {
  int i, j, elitist_archive_capacity_new, elitist_archive_size_new;
  short* elitist_archive_indices_inactive_new;
  individual** elitist_archive_new;

  if (insert_index >= elitist_archive_capacity) {
    elitist_archive_size_new = 0;
    elitist_archive_capacity_new = elitist_archive_capacity * 2 + 1;
    elitist_archive_new = (individual**)Malloc(elitist_archive_capacity_new * sizeof(individual*));
    elitist_archive_indices_inactive_new = (short*)Malloc(elitist_archive_capacity_new * sizeof(short));
    for (i = 0; i < elitist_archive_capacity_new; i++) {
      elitist_archive_new[i] = initializeIndividual();
      elitist_archive_indices_inactive_new[i] = 0;
    }

    for (i = 0; i < elitist_archive_size; i++) {
      copyIndividual(elitist_archive[i], elitist_archive_new[elitist_archive_size_new]);
      elitist_archive_size_new++;
    }

    for (i = 0; i < elitist_archive_capacity; i++)
      ezilaitiniIndividual(elitist_archive[i]);
    free(elitist_archive);
    free(elitist_archive_indices_inactive);

    elitist_archive_size = elitist_archive_size_new;
    elitist_archive_capacity = elitist_archive_capacity_new;
    elitist_archive = elitist_archive_new;
    elitist_archive_indices_inactive = elitist_archive_indices_inactive_new;
    insert_index = elitist_archive_size;
  }

  copyIndividual(ind, elitist_archive[insert_index]);

  if (insert_index == elitist_archive_size)
    elitist_archive_size++;
  elitist_archive_indices_inactive[insert_index] = 0;

  if (ind->constraint_value == 0)
    for (j = 0; j < number_of_objectives; j++)
      if (ind->objective_values[j] < best_objective_values_in_elitist_archive[j])
        best_objective_values_in_elitist_archive[j] = ind->objective_values[j];
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Output =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/**
 * Writes (appends) statistics about the current generation to a
 * file named "statistics.dat".
 */
void writeGenerationalStatisticsForOnePopulation(int population_index) {
  if (use_vtr && haveDPFSMetric())
    writeGenerationalStatisticsForOnePopulationWithDPFSMetric(population_index);
  else
    writeGenerationalStatisticsForOnePopulationWithoutDPFSMetric(population_index);
}

/**
 * Writes (appends) statistics about the current generation
 * in case of multiple objectives and the availability of the
 * D_{Pf->S} metric.
 */
void writeGenerationalStatisticsForOnePopulationWithDPFSMetric(int population_index) {
  int i, default_front_size;
  double **default_front, metric_approximation_set;
  char string[1000];
  short enable_hyper_volume;
  FILE* file;
  short* to_be_removed_solution;

  default_front = getDefaultFront(&default_front_size);

  to_be_removed_solution = (short*)Malloc(approximation_set_size * sizeof(short));
  for (i = 0; i < approximation_set_size; i++)
    to_be_removed_solution[i] = 0;

  metric_approximation_set = computeDPFSMetric(default_front, default_front_size, approximation_set,
                                               approximation_set_size, to_be_removed_solution);

  if (metric_approximation_set <= vtr)
    approximation_set_reaches_vtr = 1;

  enable_hyper_volume = 1;
  file = NULL;
  if (total_number_of_generations == 0 && statistics_file_existed == 0) {
    file = fopen("statistics.dat", "w");

    sprintf(string, "# Generation  Evaluations  Time (s)");
    fputs(string, file);
    for (i = 0; i < number_of_objectives; i++) {
      sprintf(string, "  Best_obj[%d]", i);
      fputs(string, file);
    }
    if (enable_hyper_volume) {
      sprintf(string, "  Hypervolume(approx. set)");
      fputs(string, file);
    }
    sprintf(string, "  D_{Pf->S}(approx. set)");
    fputs(string, file);
    sprintf(string, "  [ Pop.index  Subgen.  Pop.size  ]  #Real.Evals\n");
    fputs(string, file);
    statistics_file_existed = 1;
  } else
    file = fopen("statistics.dat", "a");

  sprintf(string, "  %10d %11d %11.3f", total_number_of_generations, (int)number_of_evaluations, getTimer());
  fputs(string, file);

  for (i = 0; i < number_of_objectives; i++) {
    sprintf(string, " %11.3e", best_objective_values_in_elitist_archive[i]);
    fputs(string, file);
  }

  if (enable_hyper_volume) {
    sprintf(string, " %11.3e", compute2DHyperVolume(approximation_set, approximation_set_size));
    fputs(string, file);
  }
  sprintf(string, " %11.3e", metric_approximation_set);
  fputs(string, file);

  sprintf(string, "  [ %4d %6d %10d ]  %11ld\n", population_index, number_of_generations[population_index],
          population_sizes[population_index], number_of_full_evaluations);
  fputs(string, file);

  fclose(file);
  free(to_be_removed_solution);
}

/**
 * Writes (appends) statistics about the current generation
 * in case of multiple objectives when the D_{Pf->S} metric
 * cannot be computed for the selected objective functions.
 */
void writeGenerationalStatisticsForOnePopulationWithoutDPFSMetric(int population_index) {
  int i;
  char string[1000];
  short enable_hyper_volume;
  FILE* file;

  enable_hyper_volume = 1;
  file = NULL;
  // if( total_number_of_generations == 0 && statistics_file_existed == 0 )
  if (statistics_file_existed == 0) {
    file = fopen("statistics.dat", "w");

    sprintf(string, "# Generation  Evaluations  Time (s)");
    fputs(string, file);
    for (i = 0; i < number_of_objectives; i++) {
      sprintf(string, "  Best_obj[%d]", i);
      fputs(string, file);
    }
    if (enable_hyper_volume) {
      sprintf(string, "  Hypervolume(approx. set)");
      fputs(string, file);
    }
    sprintf(string, "  [ Pop.index  Subgen.  Pop.size  ]  #Real.Evals\n");
    fputs(string, file);
    statistics_file_existed = 1;
  } else
    file = fopen("statistics.dat", "a");

  sprintf(string, "  %10d %11d %11.3f", total_number_of_generations, (int)number_of_evaluations, getTimer());
  fputs(string, file);

  for (i = 0; i < number_of_objectives; i++) {
    sprintf(string, " %11.3e", best_objective_values_in_elitist_archive[i]);
    fputs(string, file);
  }

  if (enable_hyper_volume) {
    sprintf(string, " %11.3e", compute2DHyperVolume(approximation_set, approximation_set_size));
    fputs(string, file);
  }

  sprintf(string, "  [ %4d %6d %10d ]  %11ld\n", population_index, number_of_generations[population_index],
          population_sizes[population_index], number_of_full_evaluations);
  fputs(string, file);

  fclose(file);
}

/**
 * Writes the solutions to various files. The filenames
 * contain the generation counter. If the flag final is
 * set (final != 0), the generation number in the filename
 * is replaced with the word "final".
 *
 * approximation_set_generation_xxxxx.dat: the approximation set (actual answer)
 * elitist_archive_generation_xxxxx.dat  : the elitist archive
 * population_generation_xxxxx.dat       : the population
 * selection_generation_xxxxx.dat        : the selected solutions
 * cluster_xxxxx_generation_xxxxx.dat    : the individual clusters
 */
void writeGenerationalSolutions(short final) {
  int i, j, population_index;
  char string[1000];
  FILE* file;

  // Approximation set
  if (final)
    sprintf(string, "approximation_set_generation_final.dat");
  else
    sprintf(string, "approximation_set_generation_%05d.dat", total_number_of_generations);
  file = fopen(string, "w");

  for (i = 0; i < approximation_set_size; i++) {
    for (j = 0; j < number_of_parameters; j++) {
      sprintf(string, "%13e", approximation_set[i]->parameters[j]);
      fputs(string, file);
      if (j < number_of_parameters - 1) {
        sprintf(string, " ");
        fputs(string, file);
      }
    }
    sprintf(string, "     ");
    fputs(string, file);

    for (j = 0; j < number_of_objectives; j++) {
      sprintf(string, "%13e ", approximation_set[i]->objective_values[j]);
      fputs(string, file);
    }

    sprintf(string, "%13e ", approximation_set[i]->constraint_value);
    fputs(string, file);

    sprintf(string, "\n");
    fputs(string, file);
  }

  fclose(file);

  /*
// Elitist archive
if( final )
  sprintf( string, "elitist_archive_generation_final.dat" );
else
  sprintf( string, "elitist_archive_generation_%05d.dat", total_number_of_generations );
file = fopen( string, "w" );

for( i = 0; i < elitist_archive_size; i++ )
{
  for( j = 0; j < number_of_parameters; j++ )
  {
    sprintf( string, "%13e", elitist_archive[i][j] );
    fputs( string, file );
    if( j < number_of_parameters-1 )
    {
      sprintf( string, " " );
      fputs( string, file );
    }
  }
  sprintf( string, "     " );
  fputs( string, file );
  for( j = 0; j < number_of_objectives; j++ )
  {
    sprintf( string, "%13e ", elitist_archive_objective_values[i][j] );
    fputs( string, file );
  }

  sprintf( string, "%13e\n", elitist_archive_constraint_values[i] );
  fputs( string, file );
}

fclose( file );*/

  // Population
  for (population_index = 0; population_index < number_of_populations; population_index++) {
    if (final)
      sprintf(string, "population_%03d_generation_final.dat", population_index);
    else
      sprintf(string, "population_%03d_generation_%05d.dat", population_index, total_number_of_generations);
    file = fopen(string, "w");

    for (i = 0; i < population_sizes[population_index]; i++) {
      for (j = 0; j < number_of_parameters; j++) {
        sprintf(string, "%13e", populations[population_index][i]->parameters[j]);
        fputs(string, file);
        if (j < number_of_parameters - 1) {
          sprintf(string, " ");
          fputs(string, file);
        }
      }
      sprintf(string, "     ");
      fputs(string, file);
      for (j = 0; j < number_of_objectives; j++) {
        sprintf(string, "%13e ", populations[population_index][i]->objective_values[j]);
        fputs(string, file);
      }

      sprintf(string, "%13e\n", populations[population_index][i]->constraint_value);
      fputs(string, file);
    }

    fclose(file);
  }

  // Selection
  /*if( total_number_of_generations > 0 && !final )
{
  sprintf( string, "selection_generation_%05d.dat", total_number_of_generations );
  file = fopen( string, "w" );

  for( i = 0; i < selection_sizes[population_index]; i++ )
  {
    for( j = 0; j < number_of_parameters; j++ )
    {
      sprintf( string, "%13e", selection[i][j] );
      fputs( string, file );
      if( j < number_of_parameters-1 )
      {
        sprintf( string, " " );
        fputs( string, file );
      }
    }
    sprintf( string, "     " );
    fputs( string, file );
    for( j = 0; j < number_of_objectives; j++ )
    {
      sprintf( string, "%13e ", objective_values_selection[population_index][i][j] );
      fputs( string, file );
    }

    sprintf( string, "%13e\n", constraint_values_selection[population_index][i] );
    fputs( string, file );
  }

  fclose( file );
}

// Clusters
int k;
if( total_number_of_generations > 0 && !final )
{
  for( i = 0; i < number_of_mixing_components[0]; i++ )
  {
    //if( single_objective_clusters[population_index][i] != 0 ) continue;
    sprintf( string, "cluster_%05d_generation_%05d.dat", i, total_number_of_generations );
    file = fopen( string, "w" );

    for( j = 0; j < population_sizes[0]; j++ )
    {
      if( cluster_index_for_population[0][j] != i ) continue;
      for( k = 0; k < number_of_parameters; k++ )
      {
        sprintf( string, "%13e", populations[0][j][k] );
        fputs( string, file );
        if( k < number_of_parameters-1 )
        {
          sprintf( string, " " );
          fputs( string, file );
        }
      }
      sprintf( string, "     " );
      fputs( string, file );
      for( k = 0; k < number_of_objectives; k++ )
      {
        sprintf( string, "%13e ", objective_values[0][j][k] );
        fputs( string, file );
      }

      sprintf( string, "%13e\n", constraint_values_selection[0][j] );
      fputs( string, file );
    }

    fclose( file );
  }
}*/
}

/**
 * Computes the approximation set: the non-dominated solutions
 * in the population and the elitist archive combined.
 */
void computeApproximationSet(void) {
  short dominated, same_objectives;
  int i, j, k, *indices_of_rank0, *population_indices_of_rank0, rank0_size, non_dominated_size,
      population_rank0_and_elitist_archive_size, *rank0_contribution, tot_rank0_size;
  double **population_rank0_and_elitist_archive, **population_rank0_and_elitist_archive_objective_values,
      *population_rank0_and_elitist_archive_constraint_values;

  /* First, join rank0 of the population with the elitist archive */
  indices_of_rank0 = (int*)Malloc(2 * population_sizes[number_of_populations - 1] * sizeof(int));
  population_indices_of_rank0 = (int*)Malloc(2 * population_sizes[number_of_populations - 1] * sizeof(int));
  rank0_size = 0;
  for (i = 0; i < number_of_populations; i++) {
    for (j = 0; j < population_sizes[i]; j++) {
      if (ranks[i][j] == 0) {
        indices_of_rank0[rank0_size] = j;
        population_indices_of_rank0[rank0_size] = i;
        rank0_size++;
      }
    }
  }

  population_rank0_and_elitist_archive_size = rank0_size + elitist_archive_size;
  population_rank0_and_elitist_archive = (double**)Malloc(population_rank0_and_elitist_archive_size * sizeof(double*));
  population_rank0_and_elitist_archive_objective_values =
      (double**)Malloc(population_rank0_and_elitist_archive_size * sizeof(double*));
  population_rank0_and_elitist_archive_constraint_values =
      (double*)Malloc(population_rank0_and_elitist_archive_size * sizeof(double));

  for (i = 0; i < population_rank0_and_elitist_archive_size; i++) {
    population_rank0_and_elitist_archive[i] = (double*)Malloc(number_of_parameters * sizeof(double));
    population_rank0_and_elitist_archive_objective_values[i] = (double*)Malloc(number_of_objectives * sizeof(double));
  }

  k = 0;
  for (i = 0; i < rank0_size; i++) {
    for (j = 0; j < number_of_parameters; j++)
      population_rank0_and_elitist_archive[k][j] =
          populations[population_indices_of_rank0[i]][indices_of_rank0[i]]->parameters[j];
    for (j = 0; j < number_of_objectives; j++)
      population_rank0_and_elitist_archive_objective_values[k][j] =
          populations[population_indices_of_rank0[i]][indices_of_rank0[i]]->objective_values[j];
    population_rank0_and_elitist_archive_constraint_values[k] =
        populations[population_indices_of_rank0[i]][indices_of_rank0[i]]->constraint_value;

    k++;
  }

  for (i = 0; i < elitist_archive_size; i++) {
    for (j = 0; j < number_of_parameters; j++)
      population_rank0_and_elitist_archive[k][j] = elitist_archive[i]->parameters[j];
    for (j = 0; j < number_of_objectives; j++)
      population_rank0_and_elitist_archive_objective_values[k][j] = elitist_archive[i]->objective_values[j];
    population_rank0_and_elitist_archive_constraint_values[k] = elitist_archive[i]->constraint_value;

    k++;
  }
  free(indices_of_rank0);

  /* Second, compute rank0 solutions amongst all solutions */
  indices_of_rank0 = (int*)Malloc(population_rank0_and_elitist_archive_size * sizeof(int));
  rank0_contribution = (int*)Malloc(number_of_populations * sizeof(int));
  for (i = 0; i < number_of_populations; i++)
    rank0_contribution[i] = 0;
  non_dominated_size = 0;
  for (i = 0; i < population_rank0_and_elitist_archive_size; i++) {
    dominated = 0;
    for (j = 0; j < population_rank0_and_elitist_archive_size; j++) {
      if (i != j) {
        if (constraintParetoDominates(population_rank0_and_elitist_archive_objective_values[j],
                                      population_rank0_and_elitist_archive_constraint_values[j],
                                      population_rank0_and_elitist_archive_objective_values[i],
                                      population_rank0_and_elitist_archive_constraint_values[i])) {
          dominated = 1;
          break;
        }
        same_objectives = 1;
        for (k = 0; k < number_of_objectives; k++) {
          if (population_rank0_and_elitist_archive_objective_values[i][k] !=
              population_rank0_and_elitist_archive_objective_values[j][k]) {
            same_objectives = 0;
            break;
          }
        }
        if (same_objectives &&
            (population_rank0_and_elitist_archive_constraint_values[i] ==
             population_rank0_and_elitist_archive_constraint_values[j]) &&
            (i > j)) {
          dominated = 1;
          if (i < rank0_size && j >= rank0_size)
            rank0_contribution[population_indices_of_rank0[i]]++;
          break;
        }
      }
    }

    if (!dominated) {
      if (i < rank0_size)
        rank0_contribution[population_indices_of_rank0[i]]++;
      indices_of_rank0[non_dominated_size] = i;
      non_dominated_size++;
    }
  }

  tot_rank0_size = 0;
  for (i = 0; i < number_of_populations; i++)
    tot_rank0_size += rank0_contribution[i];
  if (tot_rank0_size > 0) {
    for (i = 0; i < number_of_populations - 1; i++) {
      if (((double)rank0_contribution[i]) / (double)tot_rank0_size < 0.1)
        populations_terminated[i] = 1;
      else
        break;
    }
  }

  free(rank0_contribution);

  approximation_set_size = non_dominated_size;
  approximation_set = (individual**)Malloc(approximation_set_size * sizeof(individual*));
  for (i = 0; i < approximation_set_size; i++)
    approximation_set[i] = initializeIndividual();

  for (i = 0; i < non_dominated_size; i++) {
    for (j = 0; j < number_of_parameters; j++)
      approximation_set[i]->parameters[j] = population_rank0_and_elitist_archive[indices_of_rank0[i]][j];
    for (j = 0; j < number_of_objectives; j++)
      approximation_set[i]->objective_values[j] =
          population_rank0_and_elitist_archive_objective_values[indices_of_rank0[i]][j];
    approximation_set[i]->constraint_value =
        population_rank0_and_elitist_archive_constraint_values[indices_of_rank0[i]];
  }

  free(indices_of_rank0);
  free(population_indices_of_rank0);
  for (i = 0; i < population_rank0_and_elitist_archive_size; i++) {
    free(population_rank0_and_elitist_archive[i]);
    free(population_rank0_and_elitist_archive_objective_values[i]);
  }
  free(population_rank0_and_elitist_archive);
  free(population_rank0_and_elitist_archive_objective_values);
  free(population_rank0_and_elitist_archive_constraint_values);
}

/**
 * Frees the memory allocated for the approximation set.
 * The memory is only needed for reporting the current
 * answer (Pareto set), not for the internal workings
 * of the algorithm.
 */
void freeApproximationSet(void) {
  int i;

  for (i = 0; i < approximation_set_size; i++)
    ezilaitiniIndividual(approximation_set[i]);
  free(approximation_set);
}

/**
 * Computes the D_{Pf->S} metric for a given default front,
 * number of solutions in the default front, an approximation
 * front and the number of solutions in the approximation front.
 */
double computeDPFSMetric(double** default_front,
                         int default_front_size,
                         individual** approximation_front,
                         int approximation_front_size,
                         short* to_be_removed_solution) {
  int i, j;
  double result, distance, smallest_distance;

  if (approximation_front_size == 0)
    return (1e+308);

  result = 0.0;
  for (i = 0; i < default_front_size; i++) {
    smallest_distance = 1e+308;
    for (j = 0; j < approximation_front_size; j++) {
      if (approximation_front[j]->constraint_value == 0 && to_be_removed_solution[j] == 0) {
        distance = distanceEuclidean(default_front[i], approximation_front[j]->objective_values, number_of_objectives);
        if (distance < smallest_distance)
          smallest_distance = distance;
      }
    }
    result += smallest_distance;
  }
  result /= (double)default_front_size;

  return (result);
}

double compute2DHyperVolume(individual** pareto_front, int population_size) {
  int i, n, *sorted;
  double max_0, max_1, *obj_0, area;

  n = population_size;
  max_0 = 1.1;
  max_1 = 1.1;
  if (problem_index > 100) {
    max_0 = 0.1;
    max_1 = 0.1;
    if (problem_index == 106) {
      max_0 = 0.3;
      max_1 = 0.3;
    }
  }
  obj_0 = (double*)Malloc(n * sizeof(double));
  for (i = 0; i < n; i++)
    obj_0[i] = pareto_front[i]->objective_values[0];
  sorted = mergeSort(obj_0, n);

  area = (max_0 - fmin(max_0, obj_0[sorted[n - 1]])) *
         (max_1 - fmin(max_1, pareto_front[sorted[n - 1]]->objective_values[1]));
  for (i = n - 2; i >= 0; i--)
    area += (fmin(max_0, obj_0[sorted[i + 1]]) - fmin(max_0, obj_0[sorted[i]])) *
            (max_1 - fmin(max_1, pareto_front[sorted[i]]->objective_values[1]));

  free(obj_0);
  free(sorted);

  return area;
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Individuals -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
individual* initializeIndividual(void) {
  individual* new_individual;

  new_individual = (individual*)Malloc(sizeof(individual));
  new_individual->parameters = (double*)Malloc(number_of_parameters * sizeof(double));
  new_individual->objective_values = (double*)Malloc(number_of_objectives * sizeof(double));
  new_individual->constraint_value = 0;
  new_individual->NIS = 0;
  new_individual->parameter_sum = 0;
  return (new_individual);
}

void ezilaitiniIndividual(individual* ind) {
  if (ind != NULL) {
    if (ind->objective_values != NULL)
      free(ind->objective_values);
    if (ind->parameters != NULL)
      free(ind->parameters);
    free(ind);
  }
}

void copyIndividual(individual* source, individual* destination) {
  int i;
  for (i = 0; i < number_of_parameters; i++)
    destination->parameters[i] = source->parameters[i];
  for (i = 0; i < number_of_objectives; i++)
    destination->objective_values[i] = source->objective_values[i];
  destination->constraint_value = source->constraint_value;
  destination->parameter_sum = source->parameter_sum;
}

void copyIndividualWithoutParameters(individual* source, individual* destination) {
  int i;
  for (i = 0; i < number_of_objectives; i++)
    destination->objective_values[i] = source->objective_values[i];
  destination->constraint_value = source->constraint_value;
  destination->parameter_sum = source->parameter_sum;
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Ezilaitini -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void ezilaitiniProblem(void) {
  return;
}

};  // namespace mo_rv_gomea_impl
