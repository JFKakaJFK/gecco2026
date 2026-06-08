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

#pragma once
#ifndef _MO_RV_GOMEA_MO_OPTIMIZATION_H
#define _MO_RV_GOMEA_MO_OPTIMIZATION_H

namespace mo_rv_gomea_impl {

typedef struct individual {
  double* parameters;
  double* objective_values;
  double constraint_value;
  int NIS;

  double parameter_sum;
} individual;

};  // namespace mo_rv_gomea_impl

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "Optimization.h"
#include "FOS.h"

#include "goblin/lib/budget.h"
#include "goblin/lib/instance.h"

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace mo_rv_gomea_impl {

/*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
char* installedProblemName(int index);
int numberOfInstalledProblems(void);
int installedProblemNumberOfObjectives(int index);
double installedProblemLowerRangeBound(int index, int dimension);
double installedProblemUpperRangeBound(int index, int dimension);
void initializeParameterRangeBounds(void);
short isParameterInRangeBounds(double parameter, int dimension);
double repairParameter(double parameter, int dimension);
double distanceToRangeBounds(double* parameters);
void installedProblemEvaluation(int index,
                                individual* ind,
                                int number_of_touched_parameters,
                                int* touched_parameters_indices,
                                double* parameters_before,
                                double* objective_values_before,
                                double constraint_value_before);
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
                                               int objective_index);
void evaluateAdditionalFunctionsFull(individual* ind);
void evaluateAdditionalFunctionsPartial(individual* ind,
                                        int number_of_touched_parameters,
                                        double* touched_parameters,
                                        double* parameters_before);
void ZDT1FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index);
void ZDT1FunctionPartialProblemEvaluation(individual* ind,
                                          double* parameters,
                                          double* objective_value_result,
                                          double* constraint_value_result,
                                          int objective_index);
void ZDT1FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
void ZDT1FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
void ZDT1FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                    double* parameters,
                                                    double* objective_value_result,
                                                    double* constraint_value_result);
double ZDT1FunctionProblemLowerRangeBound(int dimension);
double ZDT1FunctionProblemUpperRangeBound(int dimension);
void ZDT2FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index);
void ZDT2FunctionPartialProblemEvaluation(individual* ind,
                                          double* parameters,
                                          double* objective_value_result,
                                          double* constraint_value_result,
                                          int objective_index);
void ZDT2FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
void ZDT2FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
void ZDT2FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                    double* parameters,
                                                    double* objective_value_result,
                                                    double* constraint_value_result);
double ZDT2FunctionProblemLowerRangeBound(int dimension);
double ZDT2FunctionProblemUpperRangeBound(int dimension);
void ZDT3FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index);
void ZDT3FunctionPartialProblemEvaluation(individual* ind,
                                          double* parameters,
                                          double* objective_value_result,
                                          double* constraint_value_result,
                                          int objective_index);
void ZDT3FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
void ZDT3FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
void ZDT3FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                    double* parameters,
                                                    double* objective_value_result,
                                                    double* constraint_value_result);
double ZDT3FunctionProblemLowerRangeBound(int dimension);
double ZDT3FunctionProblemUpperRangeBound(int dimension);
void ZDT4FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index);
void ZDT4FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
void ZDT4FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
double ZDT4FunctionProblemLowerRangeBound(int dimension);
double ZDT4FunctionProblemUpperRangeBound(int dimension);
void ZDT6FunctionProblemEvaluation(double* parameters,
                                   double* objective_value_result,
                                   double* constraint_value_result,
                                   int objective_index);
void ZDT6FunctionPartialProblemEvaluation(individual* ind,
                                          double* parameters,
                                          double* objective_value_result,
                                          double* constraint_value_result,
                                          int objective_index);
void ZDT6FunctionProblemEvaluationObjective0(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
void ZDT6FunctionProblemEvaluationObjective1(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result);
void ZDT6FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                    double* parameters,
                                                    double* objective_value_result,
                                                    double* constraint_value_result);
double ZDT6FunctionProblemLowerRangeBound(int dimension);
double ZDT6FunctionProblemUpperRangeBound(int dimension);
void BD1FunctionProblemEvaluation(double* parameters,
                                  double* objective_value_result,
                                  double* constraint_value_result,
                                  int objective_index);
void BD1FunctionProblemEvaluationObjective0(double* parameters,
                                            double* objective_value_result,
                                            double* constraint_value_result);
void BD1FunctionProblemEvaluationObjective1(double* parameters,
                                            double* objective_value_result,
                                            double* constraint_value_result);
double BD1FunctionProblemLowerRangeBound(int dimension);
double BD1FunctionProblemUpperRangeBound(int dimension);
void BD2ScaledFunctionProblemEvaluation(double* parameters,
                                        double* objective_value_result,
                                        double* constraint_value_result,
                                        int objective_index);
void BD2ScaledFunctionProblemEvaluationObjective0(double* parameters,
                                                  double* objective_value_result,
                                                  double* constraint_value_result);
void BD2ScaledFunctionProblemEvaluationObjective1(double* parameters,
                                                  double* objective_value_result,
                                                  double* constraint_value_result);
double BD2ScaledFunctionProblemLowerRangeBound(int dimension);
double BD2ScaledFunctionProblemUpperRangeBound(int dimension);
void BD2ScaledFunctionPartialProblemEvaluation(individual* ind,
                                               int number_of_touched_parameters,
                                               int* touched_parameters_indices,
                                               double* parameters_before,
                                               double* objective_values_before,
                                               double constraint_value_before,
                                               int objective_index);
void BD2ScaledFunctionPartialProblemEvaluationObjective0(individual* ind,
                                                         int number_of_touched_parameters,
                                                         int* touched_parameters_indices,
                                                         double* parameters_before,
                                                         double* objective_values_before,
                                                         double constraint_value_before);
void BD2ScaledFunctionPartialProblemEvaluationObjective1(individual* ind,
                                                         int number_of_touched_parameters,
                                                         int* touched_parameters_indices,
                                                         double* parameters_before,
                                                         double* objective_values_before,
                                                         double constraint_value_before);
void BD1FunctionPartialProblemEvaluation(individual* ind,
                                         int number_of_touched_parameters,
                                         int* touched_parameters_indices,
                                         double* parameters_before,
                                         double* objective_values_before,
                                         double constraint_value_before,
                                         int objective_index);
void BD1FunctionPartialProblemEvaluationObjective1(individual* ind,
                                                   int number_of_touched_parameters,
                                                   int* touched_parameters_indices,
                                                   double* parameters_before,
                                                   double* objective_values_before,
                                                   double constraint_value_before);
double genMED_0FunctionEvaluation(double* parameters, double exponent);
double genMED_1FunctionEvaluation(double* parameters, double exponent);
void genMEDConvex2DFunctionProblemEvaluation(double* parameters,
                                             double* objective_value_result,
                                             double* constraint_value_result,
                                             int objective_index);
void genMEDConvex2DFunctionProblemEvaluationObjective0(double* parameters,
                                                       double* objective_value_result,
                                                       double* constraint_value_result);
void genMEDConvex2DFunctionProblemEvaluationObjective1(double* parameters,
                                                       double* objective_value_result,
                                                       double* constraint_value_result);
double genMEDConvex2DFunctionProblemLowerRangeBound(int dimension);
double genMEDConvex2DFunctionProblemUpperRangeBound(int dimension);
void genMEDConcave2DFunctionProblemEvaluation(double* parameters,
                                              double* objective_value_result,
                                              double* constraint_value_result,
                                              int objective_index);
void genMEDConvex2DFunctionPartialProblemEvaluation(individual* ind,
                                                    int number_of_touched_parameters,
                                                    int* touched_parameters_indices,
                                                    double* parameters_before,
                                                    double* objective_values_before,
                                                    double constraint_value_before,
                                                    int objective_index);
void genMEDConcave2DFunctionProblemEvaluationObjective0(double* parameters,
                                                        double* objective_value_result,
                                                        double* constraint_value_result);
void genMEDConcave2DFunctionProblemEvaluationObjective1(double* parameters,
                                                        double* objective_value_result,
                                                        double* constraint_value_result);
double genMEDConcave2DFunctionProblemLowerRangeBound(int dimension);
double genMEDConcave2DFunctionProblemUpperRangeBound(int dimension);
void sumOfEllipsoidsFunctionProblemEvaluation(individual* ind,
                                              double* parameters,
                                              double* objective_value_result,
                                              double* constraint_value_result,
                                              int objective_index);
void sumOfEllipsoidsFunctionProblemEvaluationObjective0(individual* ind,
                                                        double* parameters,
                                                        double* objective_value_result,
                                                        double* constraint_value_result);
void sumOfEllipsoidsFunctionProblemEvaluationObjective1(individual* ind,
                                                        double* parameters,
                                                        double* objective_value_result,
                                                        double* constraint_value_result);
void sumOfEllipsoidsFunctionPartialProblemEvaluationObjective1(individual* ind,
                                                               double* parameters,
                                                               int number_of_touched_parameters,
                                                               int* touched_parameters_indices,
                                                               double* parameters_before,
                                                               double objective_value_before,
                                                               double constraint_value_before);
double sumOfEllipsoidsFunctionProblemLowerRangeBound(int dimension);
double sumOfEllipsoidsFunctionProblemUpperRangeBound(int dimension);
short constraintParetoDominates(double* objective_values_x,
                                double constraint_value_x,
                                double* objective_values_y,
                                double constraint_value_y);
short paretoDominates(double* objective_values_x, double* objective_values_y);
void initializeProblem(void);
void ezilaitiniProblem(void);
short haveDPFSMetric(void);
double** getDefaultFront(int* default_front_size);
double** getDefaultFrontZDT1(int* default_front_size);
double** getDefaultFrontZDT2(int* default_front_size);
double** getDefaultFrontZDT3(int* default_front_size);
double** getDefaultFrontZDT4(int* default_front_size);
double** getDefaultFrontZDT6(int* default_front_size);
double** getDefaultFrontBD1(int* default_front_size);
double** getDefaultFrontBD2(int* default_front_size);
double** getDefaultFrontBD2Scaled(int* default_front_size);
double** getDefaultFrontGenMEDConvex(int* default_front_size);
double** getDefaultFrontGenMEDConcave(int* default_front_size);
void updateElitistArchive(individual* ind);
void removeFromElitistArchive(int* indices, int number_of_indices);
void addToElitistArchive(individual* ind, int insert_index);
void adaptObjectiveDiscretization(void);
short sameObjectiveBox(double* objective_values_a, double* objective_values_b);
void writeGenerationalStatisticsForOnePopulation(int population_index);
void writeGenerationalStatisticsForOnePopulationWithDPFSMetric(int population_index);
void writeGenerationalStatisticsForOnePopulationWithoutDPFSMetric(int population_index);
void writeGenerationalSolutions(short final);
void computeApproximationSet(void);
void freeApproximationSet(void);
double computeDPFSMetric(double** default_front,
                         int default_front_size,
                         individual** approximation_front,
                         int approximation_front_size,
                         short* to_be_removed_solution);
double compute2DHyperVolume(individual** pareto_front, int population_size);
individual* initializeIndividual(void);
void ezilaitiniIndividual(individual* ind);
void copyIndividual(individual* source, individual* destination);
void copyIndividualWithoutParameters(individual* source, individual* destination);
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/

inline goblin::Rng* global_rng_ptr = NULL;
inline goblin::InstanceBase* global_problem_ptr = NULL;
inline goblin::ArchiveBase* global_archive_ptr = NULL;
inline goblin::TerminationStatus global_status{};
inline goblin::AoSSet global_solution_set{};
inline goblin::AoSSet global_parent_set{};

inline int number_of_objectives, current_population_index,
    approximation_set_size; /* Number of solutions in the final answer (the approximation set). */
inline double sum_of_ellipsoids_normalization_factor;
inline long number_of_full_evaluations;
inline short approximation_set_reaches_vtr, statistics_file_existed;
inline short objective_discretization_in_effect, /* Whether the objective space is currently being discretized for the
                                             elitist archive. */
    *elitist_archive_indices_inactive;           /* Elitist archive solutions flagged for removal. */
inline int elitist_archive_size,                 /* Number of solutions in the elitist archive. */
    elitist_archive_size_target,                 /* The lower bound of the targeted size of the elitist archive. */
    elitist_archive_capacity;                    /* Current memory allocation to elitist archive. */
inline double *best_objective_values_in_elitist_archive, /* The best objective values in the archive in the individual
                                                     objectives. */
    *objective_discretization,    /* The length of the objective discretization in each dimension (for the elitist
                                     archive). */
    **ranks;                      /* Ranks of all solutions in all populations. */
inline individual ***populations, /* The population containing the solutions. */
    ***selection,                 /* Selected solutions, one for each population. */
    **elitist_archive,            /* Archive of elitist solutions. */
    **approximation_set;          /* Set of non-dominated solutions from all populations and the elitist archive. */
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

};  // namespace mo_rv_gomea_impl

#endif
