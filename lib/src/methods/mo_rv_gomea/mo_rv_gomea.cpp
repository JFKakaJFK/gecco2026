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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "Tools.h"
#include "FOS.h"
#include "MO_optimization.h"

#include <cstdint>
#include <mutex>

#include "goblin/methods/mo_rv_gomea.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace mo_rv_gomea_impl {

/*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
void interpretCommandLine(int argc, char** argv);
void run(void);
void interpretCommandLine(int argc, char** argv);
void parseCommandLine(int argc, char** argv);
void parseOptions(int argc, char** argv, int* index);
void parseFOSElementSize(int* index, int argc, char** argv);
void printAllInstalledProblems(void);
void optionError(char** argv, int index);
void parseParameters(int argc, char** argv, int* index);
void printUsage(void);
void checkOptions(void);
void printVerboseOverview(void);
void initialize(void);
void initializeNewPopulation(void);
void initializeMemory(void);
void initializeNewPopulationMemory(int population_index);
void initializeCovarianceMatrices(int population_index);
void initializeDistributionMultipliers(int population_index);
void initializePopulationAndFitnessValues(int population_index);
short initializePopulationProblemSpecific(int population_index);
void initializePopulationPFVis(int population_index);
void computeRanks(int population_index);
void computeObjectiveRanges(int population_index);
short isSolutionInRangeBoundsForFOSElement(double* solution, int population_index, int cluster_index, int FOS_index);
short checkTerminationConditionAllPopulations(void);
short checkTerminationConditionOnePopulation(int population_index);
short checkNumberOfEvaluationsTerminationCondition(void);
short checkVTRTerminationCondition(void);
short checkDistributionMultiplierTerminationCondition(int population_index);
short checkTimeLimitTerminationCondition(void);
void makeSelection(int population_index);
int* completeSelectionBasedOnDiversityInLastSelectedRank(int population_index,
                                                         int start_index,
                                                         int number_to_select,
                                                         int* sorted);
int* greedyScatteredSubsetSelection(double** points,
                                    int number_of_points,
                                    int number_of_dimensions,
                                    int number_to_select);
void makePopulation(int population_index);
void estimateParameters(int population_index);
void estimateFullCovarianceMatrixML(int population_index, int cluster_index);
void initializeFOS(int population_index, int cluster_index);
FOS* learnLinkageTreeRVGOMEA(int population_index, int cluster_index);
void inheritDistributionMultipliers(FOS* new_FOS, FOS* prev_FOS, double* multipliers);
void evaluateCompletePopulation(int population_index);
void copyBestSolutionsToPopulation(int population_index, double** objective_values_selection_scaled);
void applyDistributionMultipliers(int population_index);
void generateAndEvaluateNewSolutionsToFillPopulationAndUpdateElitistArchive(int population_index);
short applyAMS(int population_index, int individual_index, int cluster_index);
void applyForcedImprovements(int population_index, int individual_index, short* improved);
void computeParametersForSampling(int population_index, int cluster_index);
short generateNewSolutionFromFOSElement(int population_index, int cluster_index, int FOS_index, int individual_index);
double* generateNewPartialSolutionFromFOSElement(int population_index, int cluster_index, int FOS_index);
void adaptDistributionMultipliers(int population_index, int cluster_index, int FOS_index);
short generationalImprovementForOneClusterForFOSElement(int population_index,
                                                        int cluster_index,
                                                        int FOS_index,
                                                        double* st_dev_ratio);
double getStDevRatioForOneClusterForFOSElement(int population_index,
                                               int cluster_index,
                                               int FOS_index,
                                               double* parameters);
short solutionWasImprovedByFOSElement(int population_index, int cluster_index, int FOS_index, int individual_index);
void ezilaitini(void);
void ezilaitiniMemory(void);
void ezilaitiniMemoryOnePopulation(int population_index);
void ezilaitiniDistributionMultipliers(int population_index);
void ezilaitiniCovarianceMatrices(int population_index);
void ezilaitiniObjectiveRotationMatrix(void);
void ezilaitiniParametersForSampling(int population_index);
void run(void);
int main(int argc, char** argv);

/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
std::mutex global_instance_mutex; /* We can only use the global variables once
                                     at the same time... */
goblin::StaticFOS* global_static_fos = NULL;

short print_verbose_overview, /* Whether to print a overview of settings (0 = no). */
    use_guidelines;           /* Whether to override parameters with guidelines (for those that exist). */
int base_population_size, maximum_number_of_populations, **cluster_index_for_population,
    *selection_sizes, /* The size of the selection. */
    *cluster_sizes,   /* The size of the clusters. */
    number_of_cluster_failures,
    **selection_indices, /* Indices of corresponding individuals in population for all selected individuals. */
    ***selection_indices_of_cluster_members,          /* The indices pertaining to the selection of cluster members. */
    ***selection_indices_of_cluster_members_previous, /* The indices pertaining to the selection of cluster members in
                                                         the previous generation. */
    **pop_indices_selected, **single_objective_clusters, **num_individuals_in_cluster,
    maximum_number_of_evaluations, /* The maximum number of evaluations. */
    number_of_subgenerations_per_population_factor, base_number_of_mixing_components,
    *number_of_mixing_components,                 /* The number of components in the mixture distribution. */
    number_of_nearest_neighbours_in_registration, /* The number of nearest neighbours to consider in cluster
                                                     registration */
    ***samples_drawn_from_normal, /* The number of samples drawn from the i-th normal in the last generation. */
    samples_current_cluster, ***out_of_bounds_draws, /* The number of draws that resulted in an out-of-bounds sample. */
    *no_improvement_stretch, /* The number of subsequent generations without an improvement while the distribution
                                multiplier is <= 1.0. */
    maximum_no_improvement_stretch, /* The maximum number of subsequent generations without an improvement while the
                                       distribution multiplier is <= 1.0. */
    **number_of_elitist_solutions_copied, /* The number of solutions from the elitist archive copied to the population.
                                           */
    **sorted_ranks;
double tau,                                 /* The selection truncation percentile (in [1/population_size,1]). */
    delta_AMS,                              /* The adaptation length for AMS (anticipated mean shift). */
    **objective_ranges,                     /* Ranges of objectives observed in the current population. */
    ***objective_values_selection_previous, /* Objective values of selected solutions in the previous generation,
                                               required for cluster registration. */
    **ranks_selection,                      /* Ranks of the selected solutions. */
    ***distribution_multipliers,            /* Distribution multipliers (AVS mechanism) */
    distribution_multiplier_increase,       /* The multiplicative distribution multiplier increase. */
    distribution_multiplier_decrease,       /* The multiplicative distribution multiplier decrease. */
    st_dev_ratio_threshold, /* The maximum ratio of the distance of the average improvement to the mean compared to the
                               distance of one standard deviation before triggering AVS (SDR mechanism). */
    ***mean_vectors,        /* The mean vectors, one for each population. */
    ***mean_vectors_previous,  /* The mean vectors of the previous generation, one for each population. */
    ***objective_means_scaled, /* The means of the clusters in the objective space, linearly scaled according to the
                                  observed ranges. */
    *****decomposed_covariance_matrices,             /* The covariance matrices to be used for the sampling. */
    *****decomposed_cholesky_factors_lower_triangle, /* The unique lower triangular matrix of the Cholesky factorization
                                                        for every FOS element. */
    ****full_covariance_matrix, maximum_number_of_seconds;
clock_t start, end;
FOS*** linkage_model;
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
short use_boundary_repair, /* Repair out of bound parameter value to its nearest boundary value */
    use_forced_improvement;
short copying_solutions_from_archive = 0;
int number_of_dose_points_factor;
short fix_random_seed_for_set_of_dose_calc_points, fix_random_seed_for_the_algorithm;
int64_t alg_random_seed, dose_calc_points_random_seed;

/*-=-=-=-=-=-=-=-=-=-=- Section Interpret Command Line -=-=-=-=-=-=-=-=-=-=-*/
/**
 * Parses and checks the command line.
 */
void interpretCommandLine(int argc, char** argv) {
  startTimer();

  parseCommandLine(argc, argv);

  number_of_objectives = installedProblemNumberOfObjectives(problem_index);
  if (use_guidelines) {
    tau = 0.35;
    if (maximum_number_of_populations == 1) {
      base_number_of_mixing_components = 20;
      base_population_size =
          (int)((0.5 * base_number_of_mixing_components) * (36.1 + 7.58 * log2((double)number_of_parameters)));
      // base_population_size           = (int) (0.5*((double) base_number_of_mixing_components)*(10.0*pow((double)
      // number_of_parameters,0.5))); base_population_size           = (int) (0.5*base_number_of_mixing_components*(17.0
      // + 3.0*pow((double) number_of_parameters,1.5)));
    } else {
      base_number_of_mixing_components = 1 + number_of_objectives;
      base_population_size = 10 * base_number_of_mixing_components;
    }
    distribution_multiplier_decrease = 0.9;
    st_dev_ratio_threshold = 1.0;
    maximum_no_improvement_stretch =
        (int)(2.0 + ((double)(25 + number_of_parameters)) / ((double)base_number_of_mixing_components));
  }
  statistics_file_existed = 0;
  objective_discretization_in_effect = 0;
  block_size = number_of_parameters;
  block_start = 0;
  if (problem_index == 9) {
    block_size = 5;
    block_start = 1;
  }
  number_of_blocks = (number_of_parameters - 1 + block_size - 1) / block_size;

  FOS_element_ub = number_of_parameters;
  if (FOS_element_size == -1)
    FOS_element_size = number_of_parameters;
  if (FOS_element_size == -2)
    learn_linkage_tree = 1;
  if (FOS_element_size == -3)
    static_linkage_tree = 1;
  if (FOS_element_size == -4) {
    static_linkage_tree = 1;
    FOS_element_ub = 100;
  }
  if (FOS_element_size == -5) {
    random_linkage_tree = 1;
    static_linkage_tree = 1;
    FOS_element_ub = 100;
  }
  if (FOS_element_size == 1)
    use_univariate_FOS = 1;

  checkOptions();
}

/**
 * Parses the command line.
 * For options, see printUsage.
 */
void parseCommandLine(int argc, char** argv) {
  int index;

  index = 1;

  parseOptions(argc, argv, &index);

  parseParameters(argc, argv, &index);
}

/**
 * Parses only the options from the command line.
 */
void parseOptions(int argc, char** argv, int* index) {
  double dummy;

  write_generational_statistics = 0;
  write_generational_solutions = 0;
  print_verbose_overview = 0;
  use_vtr = 0;
  use_guidelines = 0;
  black_box_evaluations = 0;

  use_boundary_repair = 0;
  use_forced_improvement = 0;

  static_linkage_tree = 0;
  random_linkage_tree = 0;

  for (; (*index) < argc; (*index)++) {
    if (argv[*index][0] == '-') {
      /* If it is a negative number, the option part is over */
      if (sscanf(argv[*index], "%lf", &dummy) && argv[*index][1] != '\0')
        break;

      if (argv[*index][1] == '\0')
        optionError(argv, *index);
      else if (argv[*index][2] != '\0')
        optionError(argv, *index);
      else {
        switch (argv[*index][1]) {
          case 'h':
            printUsage();
            break;
          case 'P':
            printAllInstalledProblems();
            break;
          case 's':
            write_generational_statistics = 1;
            break;
          case 'w':
            write_generational_solutions = 1;
            break;
          case 'v':
            print_verbose_overview = 1;
            break;
          case 'r':
            use_vtr = 1;
            break;
          case 'g':
            use_guidelines = 1;
            break;
          case 'b':
            black_box_evaluations = 1;
            break;
          case 'f':
            parseFOSElementSize(index, argc, argv);
            break;
          case 'I':
            use_forced_improvement = 1;
            break;
          case 'B':
            use_boundary_repair = 1;
            break;
          default:
            optionError(argv, *index);
        }
      }
    } else /* Argument is not an option, so option part is over */
      break;
  }
}

void parseFOSElementSize(int* index, int argc, char** argv) {
  short noError = 1;

  (*index)++;
  noError = noError && sscanf(argv[*index], "%d", &FOS_element_size);

  if (!noError) {
    printf("Error parsing parameters.\n\n");

    printUsage();
  }
}

/**
 * Writes the names of all installed problems to the standard output.
 */
void printAllInstalledProblems(void) {
  int i, n;

  n = numberOfInstalledProblems();
  printf("Installed optimization problems:\n");
  for (i = 0; i < n; i++)
    printf("%3d: %s\n", i, installedProblemName(i));

  exit(0);
}

/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError(char** argv, int index) {
  printf("Illegal option: %s\n\n", argv[index]);

  printUsage();
}

/**
 * Parses only the EA parameters from the command line.
 */
void parseParameters(int argc, char** argv, int* index) {
  int noError;

  if ((argc - *index) != 17) {
    printf("Number of parameters is incorrect, require 17 parameters (you provided %d).\n\n", (argc - *index));

    printUsage();
  }

  noError = 1;
  noError = noError && sscanf(argv[*index + 0], "%d", &problem_index);
  noError = noError && sscanf(argv[*index + 1], "%d", &number_of_parameters);
  noError = noError && sscanf(argv[*index + 2], "%lf", &lower_user_range);
  noError = noError && sscanf(argv[*index + 3], "%lf", &upper_user_range);
  noError = noError && sscanf(argv[*index + 4], "%lf", &rotation_angle);
  noError = noError && sscanf(argv[*index + 5], "%lf", &tau);
  noError = noError && sscanf(argv[*index + 6], "%d", &base_population_size);
  noError = noError && sscanf(argv[*index + 7], "%d", &maximum_number_of_populations);
  noError = noError && sscanf(argv[*index + 8], "%d", &base_number_of_mixing_components);
  noError = noError && sscanf(argv[*index + 9], "%lf", &distribution_multiplier_decrease);
  noError = noError && sscanf(argv[*index + 10], "%lf", &st_dev_ratio_threshold);
  noError = noError && sscanf(argv[*index + 11], "%d", &elitist_archive_size_target);
  noError = noError && sscanf(argv[*index + 12], "%d", &maximum_number_of_evaluations);
  noError = noError && sscanf(argv[*index + 13], "%lf", &vtr);
  noError = noError && sscanf(argv[*index + 14], "%d", &maximum_no_improvement_stretch);
  noError = noError && sscanf(argv[*index + 15], "%lf", &maximum_number_of_seconds);
  noError = noError && sscanf(argv[*index + 16], "%d", &number_of_subgenerations_per_population_factor);

  if (!noError) {
    printf("Error parsing parameters.\n\n");

    printUsage();
  }
}

/**
 * Prints usage information and exits the program.
 */
void printUsage(void) {
  printf("Usage: RV-MO-GOMEA [-?] pro dim low upp rot tau pop nop noc dmd srt etp ets tar eva vtr imp sec sub\n");
  printf(" -h: Prints out this usage information.\n");
  printf(" -P: Prints out a list of all installed optimization problems.\n");
  printf(" -s: Enables computing and writing of statistics every generation.\n");
  printf(" -w: Enable writing of solutions and their fitnesses every generation.\n");
  printf(" -v: Verbose mode. Prints the settings before starting the run.\n");
  printf(" -r: Enables use of vtr in termination condition (value-to-reach).\n");
  printf(" -b: Enables black-box optimization (all evaluations are full evaluations).\n");
  printf(" -f %%d: Sets linkage model that is used. Positive: Use a FOS with elements of %%d consecutive variables.\n");
  printf(
      "     Use -1 for full linkage model, -2 for dynamic linkage tree learned from the population, -3 for fixed "
      "linkage tree learned from distance measure,\n");
  printf(
      "     -4 for bounded fixed linkage tree learned from distance measure, -5 for fixed bounded linkage tree learned "
      "from random distance measure.\n");
  printf(" -g: Uses guidelines to override parameter settings for those parameters\n");
  printf("     for which a guideline is known in literature. These parameters are:\n");
  printf("     tau pop dmd srt imp\n");
  printf(" -I: Enables forced improvements.\n");
  printf(" -B: Enables boundary repair.\n");
  printf("\n");
  printf("  pro: Index of optimization problem to be solved (minimization).\n");
  printf("  dim: Number of parameters.\n");
  printf("  low: Overall initialization lower bound.\n");
  printf("  upp: Overall initialization upper bound.\n");
  printf("  rot: The angle by which to rotate the problem.\n");
  printf("  tau: Selection percentile (tau in [1/pop,1], truncation selection).\n");
  printf("  pop: Base population size.\n");
  printf("  nop: Maximum number of populations.\n");
  printf("  noc: Base number of components in the mixture distribution.\n");
  printf("  dmd: The distribution multiplier decreaser (in (0,1), increaser is always 1/dmd).\n");
  printf("  srt: The standard-devation ratio threshold for triggering variance-scaling.\n");
  printf("  tar: The target number of solutions to have in the elitist archive.\n");
  printf("  eva: Maximum number of evaluations allowed.\n");
  printf("  vtr: The value to reach. If the objective value of the best feasible solution reaches\n");
  printf("       this value, termination is enforced (if -r is specified).\n");
  printf("  imp: Maximum number of subsequent generations without an improvement while the\n");
  printf("       the distribution multiplier is <= 1.0.\n");
  printf("  sec: Time limit in seconds.\n");
  printf("  sub: Subgeneration factor in interleaved multi-start scheme.\n");
  exit(0);
}

/**
 * Checks whether the selected options are feasible.
 */
void checkOptions(void) {
  if (number_of_parameters < 1) {
    printf("\n");
    printf("Error: number of parameters < 1 (read: %d). Require number of parameters >= 1.", number_of_parameters);
    printf("\n\n");

    exit(0);
  }

  if (((int)(tau * base_population_size)) <= 0 || tau >= 1) {
    printf("\n");
    printf("Error: tau not in range (read: %e). Require tau in [1/pop,1] (read: [%e,%e]).", tau,
           1.0 / ((double)base_population_size), 1.0);
    printf("\n\n");

    exit(0);
  }

  if (base_population_size < 1) {
    printf("\n");
    printf("Error: population size < 1 (read: %d). Require population size >= 1.", base_population_size);
    printf("\n\n");

    exit(0);
  }

  if (base_number_of_mixing_components < 1) {
    printf("\n");
    printf("Error: number of mixing components < 1 (read: %d). Require number of mixture components >= 1.",
           base_number_of_mixing_components);
    printf("\n\n");

    exit(0);
  }

  if (elitist_archive_size_target < 1) {
    printf("\n");
    printf("Error: elitist archive size target < 1 (read: %d).", elitist_archive_size_target);
    printf("\n\n");

    exit(0);
  }

  if (installedProblemName(problem_index) == NULL) {
    printf("\n");
    printf("Error: unknown index for problem (read index %d).", problem_index);
    printf("\n\n");

    exit(0);
  }

  if (rotation_angle > 0 && (!learn_linkage_tree && FOS_element_size >= 0 && FOS_element_size != block_size &&
                             FOS_element_size != number_of_parameters)) {
    printf("\n");
    printf("Error: invalid FOS element size (read %d). Must be %d, %d or %d.", FOS_element_size, 1, block_size,
           number_of_parameters);
    printf("\n\n");

    exit(0);
  }
}

/**
 * Prints the settings as read from the command line.
 */
void printVerboseOverview(void) {
  int i;

  printf("### Settings ######################################\n");
  printf("#\n");
  printf("# Statistics writing every generation: %s\n", write_generational_statistics ? "enabled" : "disabled");
  printf("# Population file writing            : %s\n", write_generational_solutions ? "enabled" : "disabled");
  printf("# Use of value-to-reach (vtr)        : %s\n", use_vtr ? "enabled" : "disabled");
  printf("#\n");
  printf("###################################################\n");
  printf("#\n");
  printf("# Problem                  = %s\n", installedProblemName(problem_index));
  printf("# Number of parameters     = %d\n", number_of_parameters);
  printf("# Initialization ranges    = ");
  for (i = 0; i < number_of_parameters; i++) {
    printf("x_%d: [%e;%e]", i, lower_init_ranges[i], upper_init_ranges[i]);
    if (i < number_of_parameters - 1)
      printf("\n#                            ");
  }
  printf("\n");
  printf("# Boundary ranges          = ");
  for (i = 0; i < number_of_parameters; i++) {
    printf("x_%d: [%e;%e]", i, lower_range_bounds[i], upper_range_bounds[i]);
    if (i < number_of_parameters - 1)
      printf("\n#                            ");
  }
  printf("\n");
  printf("# Rotation angle           = %e\n", rotation_angle);
  printf("# Tau                      = %e\n", tau);
  printf("# Population size          = %d\n", base_population_size);
  printf("# Number of populations    = %d\n", maximum_number_of_populations);
  printf("# FOS element size         = %d\n", FOS_element_size);
  printf("# Number of mix. com. (k)  = %d\n", base_number_of_mixing_components);
  printf("# Dis. mult. decreaser     = %e\n", distribution_multiplier_decrease);
  printf("# St. dev. rat. threshold  = %e\n", st_dev_ratio_threshold);
  printf("# Elitist ar. size target  = %d\n", elitist_archive_size_target);
  printf("# Maximum numb. of eval.   = %d\n", maximum_number_of_evaluations);
  printf("# Value to reach (vtr)     = %e\n", vtr);
  printf("# Time limit (s)           = %e\n", maximum_number_of_seconds);
  printf("# Random seed              = %ld\n", (long)random_seed);
  printf("#\n");
  printf("###################################################\n");
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Initialize -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Performs initializations that are required before starting a run.
 */
void initialize(void) {
  int i, j;

  number_of_populations = 0;
  total_number_of_generations = 0;
  number_of_evaluations = 0;
  number_of_full_evaluations = 0;
  approximation_set_reaches_vtr = 0;
  distribution_multiplier_increase = 1.0 / distribution_multiplier_decrease;

  initializeProblem();

  for (i = 1; i < number_of_parameters; i++) {
    j = (i - 1) % block_size;
    sum_of_ellipsoids_normalization_factor += pow(10.0, 6.0 * (((double)(j)) / ((double)(block_size - 1))));
  }
  //
  delta_AMS = 2.0;
  if (problem_index < 9 && static_linkage_tree)
    random_linkage_tree = 1;
  initializeMemory();

  initializeParameterRangeBounds();

  initializeObjectiveRotationMatrix();
}

void initializeNewPopulation(void) {
  current_population_index = number_of_populations;

  initializeNewPopulationMemory(number_of_populations);

  initializePopulationAndFitnessValues(number_of_populations);

  if (!learn_linkage_tree) {
    initializeCovarianceMatrices(number_of_populations);

    initializeDistributionMultipliers(number_of_populations);
  }

  computeObjectiveRanges(number_of_populations);

  computeRanks(number_of_populations);

  number_of_populations++;
}

void initializeMemory() {
  int i;

  elitist_archive_size = 0;
  elitist_archive_capacity = 10;
  populations = (individual***)Malloc(maximum_number_of_populations * sizeof(individual**));
  selection = (individual***)Malloc(maximum_number_of_populations * sizeof(individual**));
  population_sizes = (int*)Malloc(maximum_number_of_populations * sizeof(int));
  populations_terminated = (short*)Malloc(maximum_number_of_populations * sizeof(short));
  selection_sizes = (int*)Malloc(maximum_number_of_populations * sizeof(int));
  cluster_sizes = (int*)Malloc(maximum_number_of_populations * sizeof(int));
  cluster_index_for_population = (int**)Malloc(maximum_number_of_populations * sizeof(int*));
  ranks = (double**)Malloc(maximum_number_of_populations * sizeof(double*));
  sorted_ranks = (int**)Malloc(maximum_number_of_populations * sizeof(int*));
  objective_ranges = (double**)Malloc(maximum_number_of_populations * sizeof(double*));
  selection_indices = (int**)Malloc(maximum_number_of_populations * sizeof(int*));
  objective_values_selection_previous = (double***)Malloc(maximum_number_of_populations * sizeof(double**));
  ranks_selection = (double**)Malloc(maximum_number_of_populations * sizeof(double*));
  pop_indices_selected = (int**)Malloc(maximum_number_of_populations * sizeof(int*));
  number_of_elitist_solutions_copied = (int**)Malloc(maximum_number_of_populations * sizeof(int*));
  objective_means_scaled = (double***)Malloc(maximum_number_of_populations * sizeof(double**));
  mean_vectors = (double***)Malloc(maximum_number_of_populations * sizeof(double**));
  mean_vectors_previous = (double***)Malloc(maximum_number_of_populations * sizeof(double**));
  decomposed_cholesky_factors_lower_triangle = (double*****)Malloc(maximum_number_of_populations * sizeof(double****));
  selection_indices_of_cluster_members = (int***)Malloc(maximum_number_of_populations * sizeof(int**));
  selection_indices_of_cluster_members_previous = (int***)Malloc(maximum_number_of_populations * sizeof(int**));
  single_objective_clusters = (int**)Malloc(maximum_number_of_populations * sizeof(int*));
  num_individuals_in_cluster = (int**)Malloc(maximum_number_of_populations * sizeof(int*));
  number_of_mixing_components = (int*)Malloc(maximum_number_of_populations * sizeof(int));
  distribution_multipliers = (double***)Malloc(maximum_number_of_populations * sizeof(double**));
  decomposed_covariance_matrices = (double*****)Malloc(maximum_number_of_populations * sizeof(double****));
  full_covariance_matrix = (double****)Malloc(maximum_number_of_populations * sizeof(double***));
  samples_drawn_from_normal = (int***)Malloc(maximum_number_of_populations * sizeof(int**));
  out_of_bounds_draws = (int***)Malloc(maximum_number_of_populations * sizeof(int**));
  number_of_generations = (int*)Malloc(maximum_number_of_populations * sizeof(int));
  no_improvement_stretch = (int*)Malloc(maximum_number_of_populations * sizeof(int));
  linkage_model = (FOS***)Malloc(maximum_number_of_populations * sizeof(FOS**));

  objective_discretization = (double*)Malloc(number_of_objectives * sizeof(double));
  elitist_archive = (individual**)Malloc(elitist_archive_capacity * sizeof(individual*));
  best_objective_values_in_elitist_archive = (double*)Malloc(number_of_objectives * sizeof(double));
  elitist_archive_indices_inactive = (short*)Malloc(elitist_archive_capacity * sizeof(short));

  for (i = 0; i < elitist_archive_capacity; i++) {
    elitist_archive[i] = initializeIndividual();
    elitist_archive_indices_inactive[i] = 0;
  }

  for (i = 0; i < number_of_objectives; i++) {
    best_objective_values_in_elitist_archive[i] = 1e+308;
  }

  for (i = 0; i < maximum_number_of_populations; i++) {
    distribution_multipliers[i] = NULL;
    samples_drawn_from_normal[i] = NULL;
    out_of_bounds_draws[i] = NULL;
  }
}

/**
 * Initializes the memory.
 */
void initializeNewPopulationMemory(int population_index) {
  int i;

  if (population_index == 0) {
    population_sizes[population_index] = base_population_size;
    number_of_mixing_components[population_index] = base_number_of_mixing_components;
  } else {
    population_sizes[population_index] = 2 * population_sizes[population_index - 1];
    number_of_mixing_components[population_index] = number_of_mixing_components[population_index - 1] + 1;
  }
  selection_sizes[population_index] =
      population_sizes[population_index];  // HACK(int) (tau*(population_sizes[population_index]));
  cluster_sizes[population_index] =
      (2 * ((int)(tau * (population_sizes[population_index])))) /
      number_of_mixing_components[population_index];  // HACK(2*selection_size)/number_of_mixing_components;
  number_of_generations[population_index] = 0;
  populations_terminated[population_index] = 0;
  no_improvement_stretch[population_index] = 0;

  populations[population_index] = (individual**)Malloc(population_sizes[population_index] * sizeof(individual*));
  selection[population_index] = (individual**)Malloc(selection_sizes[population_index] * sizeof(individual*));
  cluster_index_for_population[population_index] = (int*)Malloc(population_sizes[population_index] * sizeof(int));
  ranks[population_index] = (double*)Malloc(population_sizes[population_index] * sizeof(double));
  sorted_ranks[population_index] = (int*)Malloc(population_sizes[population_index] * sizeof(int));
  objective_ranges[population_index] = (double*)Malloc(population_sizes[population_index] * sizeof(double));
  selection_indices[population_index] = (int*)Malloc(selection_sizes[population_index] * sizeof(int));
  objective_values_selection_previous[population_index] =
      (double**)Malloc(selection_sizes[population_index] * sizeof(double*));
  ranks_selection[population_index] = (double*)Malloc(selection_sizes[population_index] * sizeof(double));
  pop_indices_selected[population_index] = (int*)Malloc(population_sizes[population_index] * sizeof(int));
  number_of_elitist_solutions_copied[population_index] =
      (int*)Malloc(number_of_mixing_components[population_index] * sizeof(int));
  objective_means_scaled[population_index] =
      (double**)Malloc(number_of_mixing_components[population_index] * sizeof(double*));
  mean_vectors[population_index] = (double**)Malloc(number_of_mixing_components[population_index] * sizeof(double*));
  mean_vectors_previous[population_index] =
      (double**)Malloc(number_of_mixing_components[population_index] * sizeof(double*));
  decomposed_cholesky_factors_lower_triangle[population_index] =
      (double****)Malloc(number_of_mixing_components[population_index] * sizeof(double***));
  selection_indices_of_cluster_members[population_index] =
      (int**)Malloc(number_of_mixing_components[population_index] * sizeof(int*));
  selection_indices_of_cluster_members_previous[population_index] =
      (int**)Malloc(number_of_mixing_components[population_index] * sizeof(int*));
  single_objective_clusters[population_index] =
      (int*)Malloc(number_of_mixing_components[population_index] * sizeof(int));
  num_individuals_in_cluster[population_index] =
      (int*)Malloc(number_of_mixing_components[population_index] * sizeof(int));
  linkage_model[population_index] = (FOS**)Malloc(number_of_mixing_components[population_index] * sizeof(FOS*));

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    linkage_model[population_index][i] = NULL;

  for (i = 0; i < population_sizes[population_index]; i++)
    populations[population_index][i] = initializeIndividual();

  for (i = 0; i < selection_sizes[population_index]; i++) {
    selection[population_index][i] = initializeIndividual();

    objective_values_selection_previous[population_index][i] = (double*)Malloc(number_of_objectives * sizeof(double));
  }

  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    mean_vectors[population_index][i] = (double*)Malloc(number_of_parameters * sizeof(double));

    mean_vectors_previous[population_index][i] = (double*)Malloc(number_of_parameters * sizeof(double));

    selection_indices_of_cluster_members[population_index][i] = NULL;

    selection_indices_of_cluster_members_previous[population_index][i] = NULL;

    objective_means_scaled[population_index][i] = (double*)Malloc(number_of_objectives * sizeof(double));
  }

  if (learn_linkage_tree) {
    full_covariance_matrix[population_index] =
        (double***)Malloc(number_of_mixing_components[population_index] * sizeof(double**));
  } else {
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      initializeFOS(population_index, i);
  }
}

void initializeCovarianceMatrices(int population_index) {
  int i, j, k, m;

  decomposed_covariance_matrices[population_index] =
      (double****)Malloc(number_of_mixing_components[population_index] * sizeof(double***));
  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    decomposed_covariance_matrices[population_index][i] =
        (double***)Malloc(linkage_model[population_index][i]->length * sizeof(double**));
    for (j = 0; j < linkage_model[population_index][i]->length; j++) {
      decomposed_covariance_matrices[population_index][i][j] =
          (double**)Malloc(linkage_model[population_index][i]->set_length[j] * sizeof(double*));
      for (k = 0; k < linkage_model[population_index][i]->set_length[j]; k++) {
        decomposed_covariance_matrices[population_index][i][j][k] =
            (double*)Malloc(linkage_model[population_index][i]->set_length[j] * sizeof(double));
        for (m = 0; m < linkage_model[population_index][i]->set_length[j]; m++) {
          decomposed_covariance_matrices[population_index][i][j][k][m] = 1;
        }
      }
    }
  }
}

/**
 * Initializes the distribution multipliers.
 */
void initializeDistributionMultipliers(int population_index) {
  int i, j;

  distribution_multipliers[population_index] =
      (double**)Malloc(number_of_mixing_components[population_index] * sizeof(double*));
  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    distribution_multipliers[population_index][i] =
        (double*)Malloc(linkage_model[population_index][i]->length * sizeof(double));
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
      distribution_multipliers[population_index][i][j] = 1.0;
  }

  samples_drawn_from_normal[population_index] =
      (int**)Malloc(number_of_mixing_components[population_index] * sizeof(int*));
  out_of_bounds_draws[population_index] = (int**)Malloc(number_of_mixing_components[population_index] * sizeof(int*));
  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    samples_drawn_from_normal[population_index][i] =
        (int*)Malloc(linkage_model[population_index][i]->length * sizeof(int));
    out_of_bounds_draws[population_index][i] = (int*)Malloc(linkage_model[population_index][i]->length * sizeof(int));
  }
}

short initializePopulationProblemSpecific(int population_index) {
  switch (problem_index) {
    default:
      return (0);
  }
}

/**
 * Initializes the population and the fitness values.
 */
void initializePopulationAndFitnessValues(int population_index) {
  int i, j;
  short problem_specific_initialize;

  problem_specific_initialize = initializePopulationProblemSpecific(population_index);
  if (!problem_specific_initialize) {
    for (i = 0; i < population_sizes[population_index]; i++)
      for (j = 0; j < number_of_parameters; j++)
        populations[population_index][i]->parameters[j] =
            lower_init_ranges[j] + (upper_init_ranges[j] - lower_init_ranges[j]) * randomRealUniform01();
  }

  for (i = 0; i < population_sizes[population_index]; i++) {
    installedProblemEvaluation(problem_index, populations[population_index][i], number_of_parameters, NULL, NULL, NULL,
                               0);

    updateElitistArchive(populations[population_index][i]);

    if (global_terminate_immediately) {
      return;
    }
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Ranking -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Computes the ranks of the solutions in all populations.
 */
void computeRanks(int population_index) {
  short **domination_matrix, is_illegal;
  int i, j, k, *being_dominated_count, rank, number_of_solutions_ranked, *indices_in_this_rank;

  for (i = 0; i < population_sizes[population_index]; i++) {
    is_illegal = 0;
    for (j = 0; j < number_of_objectives; j++) {
      if (isnan(populations[population_index][i]->objective_values[j])) {
        is_illegal = 1;
        break;
      }
    }
    if (isnan(populations[population_index][i]->constraint_value))
      is_illegal = 1;

    if (is_illegal) {
      for (j = 0; j < number_of_objectives; j++)
        populations[population_index][i]->objective_values[j] = 1e+308;
      populations[population_index][i]->constraint_value = 1e+308;
    }
  }

  /* The domination matrix stores for each solution i
   * whether it dominates solution j, i.e. domination[i][j] = 1. */
  domination_matrix = (short**)Malloc(population_sizes[population_index] * sizeof(short*));
  for (i = 0; i < population_sizes[population_index]; i++)
    domination_matrix[i] = (short*)Malloc(population_sizes[population_index] * sizeof(short));

  being_dominated_count = (int*)Malloc(population_sizes[population_index] * sizeof(int));

  for (i = 0; i < population_sizes[population_index]; i++) {
    being_dominated_count[i] = 0;
    for (j = 0; j < population_sizes[population_index]; j++)
      domination_matrix[i][j] = 0;
  }

  for (i = 0; i < population_sizes[population_index]; i++) {
    for (j = 0; j < population_sizes[population_index]; j++) {
      if (i != j) {
        if (constraintParetoDominates(populations[population_index][i]->objective_values,
                                      populations[population_index][i]->constraint_value,
                                      populations[population_index][j]->objective_values,
                                      populations[population_index][j]->constraint_value)) {
          domination_matrix[i][j] = 1;
          being_dominated_count[j]++;
        }
      }
    }
  }

  /* Compute ranks from the domination matrix */
  rank = 0;
  number_of_solutions_ranked = 0;
  indices_in_this_rank = (int*)Malloc(population_sizes[population_index] * sizeof(int));
  while (number_of_solutions_ranked < population_sizes[population_index]) {
    k = 0;
    for (i = 0; i < population_sizes[population_index]; i++) {
      if (being_dominated_count[i] == 0) {
        ranks[population_index][i] = rank;
        indices_in_this_rank[k] = i;
        k++;
        being_dominated_count[i]--;
        number_of_solutions_ranked++;
      }
    }

    for (i = 0; i < k; i++) {
      for (j = 0; j < population_sizes[population_index]; j++) {
        if (domination_matrix[indices_in_this_rank[i]][j] == 1)
          being_dominated_count[j]--;
      }
    }

    rank++;
  }

  free(indices_in_this_rank);

  free(being_dominated_count);

  for (i = 0; i < population_sizes[population_index]; i++)
    free(domination_matrix[i]);
  free(domination_matrix);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Output =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Computes the ranges of all fitness values
 * of all solutions currently in the populations.
 */
void computeObjectiveRanges(int population_index) {
  int i, j;
  double low, high;

  for (j = 0; j < number_of_objectives; j++) {
    low = 1e+308;
    high = -1e+308;

    for (i = 0; i < population_sizes[population_index]; i++) {
      if (populations[population_index][i]->objective_values[j] < low)
        low = populations[population_index][i]->objective_values[j];
      if (populations[population_index][i]->objective_values[j] > high &&
          (populations[population_index][i]->objective_values[j] <= 1e+300))
        high = populations[population_index][i]->objective_values[j];
    }

    objective_ranges[population_index][j] = high - low;
  }
}

/**
 * Returns whether a solution is inside the range bound of
 * every objective function.
 */
short isSolutionInRangeBoundsForFOSElement(double* solution, int population_index, int cluster_index, int FOS_index) {
  int i;

  for (i = 0; i < linkage_model[population_index][cluster_index]->set_length[FOS_index]; i++)
    if (!isParameterInRangeBounds(solution[linkage_model[population_index][cluster_index]->sets[FOS_index][i]],
                                  linkage_model[population_index][cluster_index]->sets[FOS_index][i]))
      return (0);

  return (1);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Termination -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns 1 if termination should be enforced, 0 otherwise.
 */
short checkTerminationConditionAllPopulations(void) {
  int i;

  if (global_terminate_immediately) {
    return 1;
  }

  if (number_of_populations == 0)
    return (0);

  if (checkNumberOfEvaluationsTerminationCondition()) {
    global_status = goblin::TerminationStatus::EvaluationLimitReached;
    return (1);
  }

  // if (checkVTRTerminationCondition())
  //   return (1);

  if (checkTimeLimitTerminationCondition()) {
    global_status = goblin::TerminationStatus::TimeLimitReached;
    return (1);
  }

  for (i = 0; i < number_of_populations; i++)
    if (checkDistributionMultiplierTerminationCondition(i))
      populations_terminated[i] = 1;

  return (0);
}

short checkTerminationConditionOnePopulation(int population_index) {
  if (global_terminate_immediately) {
    return 1;
  }

  if (number_of_populations == 0)
    return (0);

  if (checkNumberOfEvaluationsTerminationCondition()) {
    global_status = goblin::TerminationStatus::EvaluationLimitReached;
    return (1);
  }

  // if (checkVTRTerminationCondition())
  //   return (1);

  if (checkTimeLimitTerminationCondition()) {
    global_status = goblin::TerminationStatus::TimeLimitReached;
    return (1);
  }

  if (checkDistributionMultiplierTerminationCondition(population_index))
    populations_terminated[population_index] = 1;

  return (0);
}

/**
 * Returns 1 if the maximum number of evaluations
 * has been reached, 0 otherwise.
 */
short checkNumberOfEvaluationsTerminationCondition(void) {
  if (number_of_evaluations >= maximum_number_of_evaluations && maximum_number_of_evaluations > 0)
    return (1);

  return (0);
}

/**
 * Returns 1 if the value-to-reach has been reached
 * for the multi-objective case. This means that
 * the D_Pf->S metric has reached the value-to-reach.
 * If no D_Pf->S can be computed, 0 is returned.
 */
short checkVTRTerminationCondition(void) {
  int default_front_size;
  double **default_front, metric_elitist_archive;

  if (use_vtr && haveDPFSMetric()) {
    if (approximation_set_reaches_vtr)
      return (1);

    default_front = getDefaultFront(&default_front_size);
    metric_elitist_archive = computeDPFSMetric(default_front, default_front_size, elitist_archive, elitist_archive_size,
                                               elitist_archive_indices_inactive);

    if (metric_elitist_archive <= vtr)
      return (1);
  }

  return (0);
}

/**
 * Checks whether the distribution multiplier for any mixture component
 * has become too small (0.5).
 */
short checkDistributionMultiplierTerminationCondition(int population_index) {
  int i, j;

  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
      if (distribution_multipliers[population_index][i][j] > 5e-1)
        return (0);
  }

  return (1);
}

short checkTimeLimitTerminationCondition(void) {
  if (maximum_number_of_seconds > 0 && getTimer() > maximum_number_of_seconds)
    return (1);
  return (0);
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Selection =-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Makes a set of selected solutions by taking the solutions from all
 * ranks up to the rank of the solution at the selection-size boundary.
 * The selection is filled using a diverse selection from the final rank.
 */
void makeSelection(int population_index) {
  int i, j, k, individuals_selected, individuals_to_select, last_selected_rank, elitist_solutions_copied;

  for (i = 0; i < selection_sizes[population_index]; i++)
    for (j = 0; j < number_of_objectives; j++)
      objective_values_selection_previous[population_index][i][j] = selection[population_index][i]->objective_values[j];

  for (i = 0; i < population_sizes[population_index]; i++)
    pop_indices_selected[population_index][i] = -1;

  free(sorted_ranks[population_index]);
  sorted_ranks[population_index] = mergeSort(ranks[population_index], population_sizes[population_index]);

  // Copy elitist archive to selection
  elitist_solutions_copied = 0;
  individuals_selected = elitist_solutions_copied;
  individuals_to_select = ((int)(tau * population_sizes[population_index])) - elitist_solutions_copied;
  last_selected_rank = (int)ranks[population_index][sorted_ranks[population_index][individuals_to_select - 1]];

  i = 0;
  while (((int)ranks[population_index][sorted_ranks[population_index][i]]) != last_selected_rank) {
    for (j = 0; j < number_of_parameters; j++)
      selection[population_index][individuals_selected]->parameters[j] =
          populations[population_index][sorted_ranks[population_index][i]]->parameters[j];
    for (j = 0; j < number_of_objectives; j++)
      selection[population_index][individuals_selected]->objective_values[j] =
          populations[population_index][sorted_ranks[population_index][i]]->objective_values[j];
    selection[population_index][individuals_selected]->constraint_value =
        populations[population_index][sorted_ranks[population_index][i]]->constraint_value;
    ranks_selection[population_index][individuals_selected] =
        ranks[population_index][sorted_ranks[population_index][i]];
    selection_indices[population_index][individuals_selected] = sorted_ranks[population_index][i];
    pop_indices_selected[population_index][sorted_ranks[population_index][i]] = individuals_selected;

    i++;
    individuals_selected++;
  }

  int *selected_indices, start_index;
  selected_indices = NULL;
  if (individuals_selected < individuals_to_select)
    selected_indices = completeSelectionBasedOnDiversityInLastSelectedRank(
        population_index, i, individuals_to_select - individuals_selected, sorted_ranks[population_index]);

  if (selected_indices) {
    start_index = i;
    for (j = 0; individuals_selected < individuals_to_select; individuals_selected++, j++)
      pop_indices_selected[population_index][sorted_ranks[population_index][selected_indices[j] + start_index]] =
          individuals_selected;
  }

  j = individuals_to_select;
  for (i = 0; i < population_sizes[population_index]; i++) {
    if (pop_indices_selected[population_index][i] == -1) {
      for (k = 0; k < number_of_parameters; k++)
        selection[population_index][j]->parameters[k] = populations[population_index][i]->parameters[k];
      for (k = 0; k < number_of_objectives; k++)
        selection[population_index][j]->objective_values[k] = populations[population_index][i]->objective_values[k];
      selection[population_index][j]->constraint_value = populations[population_index][i]->constraint_value;
      ranks_selection[population_index][j] = ranks[population_index][i];
      selection_indices[population_index][j] = i;
      pop_indices_selected[population_index][i] = j;
      j++;
    }
  }

  if (selected_indices)
    free(selected_indices);
}

/**
 * Fills up the selection by using greedy diversity selection
 * in the last selected rank.
 */
int* completeSelectionBasedOnDiversityInLastSelectedRank(int population_index,
                                                         int start_index,
                                                         int number_to_select,
                                                         int* sorted) {
  int i, j, *selected_indices, number_of_points, number_of_dimensions;
  double** points;

  /* Determine the solutions to select from */
  number_of_points = 0;
  while (ranks[population_index][sorted[start_index + number_of_points]] ==
         ranks[population_index][sorted[start_index]]) {
    number_of_points++;
    if ((start_index + number_of_points) == population_sizes[population_index])
      break;
  }

  points = (double**)Malloc(number_of_points * sizeof(double*));
  for (i = 0; i < number_of_points; i++)
    points[i] = (double*)Malloc(number_of_objectives * sizeof(double));
  for (i = 0; i < number_of_points; i++)
    for (j = 0; j < number_of_objectives; j++)
      points[i][j] = populations[population_index][sorted[start_index + i]]->objective_values[j] /
                     objective_ranges[population_index][j];

  /* Select */
  number_of_dimensions = number_of_objectives;
  selected_indices = greedyScatteredSubsetSelection(points, number_of_points, number_of_dimensions, number_to_select);

  /* Copy to selection */
  for (i = 0; i < number_to_select; i++) {
    for (j = 0; j < number_of_parameters; j++)
      selection[population_index][i + start_index]->parameters[j] =
          populations[population_index][sorted[selected_indices[i] + start_index]]->parameters[j];
    for (j = 0; j < number_of_objectives; j++)
      selection[population_index][i + start_index]->objective_values[j] =
          populations[population_index][sorted[selected_indices[i] + start_index]]->objective_values[j];
    selection[population_index][i + start_index]->constraint_value =
        populations[population_index][sorted[selected_indices[i] + start_index]]->constraint_value;
    ranks_selection[population_index][i + start_index] =
        ranks[population_index][sorted[selected_indices[i] + start_index]];
    selection_indices[population_index][i + start_index] = sorted[selected_indices[i] + start_index];
  }

  for (i = 0; i < number_of_points; i++)
    free(points[i]);
  free(points);

  return (selected_indices);
}

/**
 * Selects n points from a set of points. A
 * greedy heuristic is used to find a good
 * scattering of the selected points. First,
 * a point is selected with a maximum value
 * in a randomly selected dimension. The
 * remaining points are selected iteratively.
 * In each iteration, the point selected is
 * the one that maximizes the minimal distance
 * to the points selected so far.
 */
int* greedyScatteredSubsetSelection(double** points,
                                    int number_of_points,
                                    int number_of_dimensions,
                                    int number_to_select) {
  int i, index_of_farthest, random_dimension_index, number_selected_so_far, *indices_left, *result;
  double *nn_distances, distance_of_farthest, value;

  if (number_to_select > number_of_points) {
    printf("\n");
    printf("Error: greedyScatteredSubsetSelection asked to select %d solutions from set of size %d.", number_to_select,
           number_of_points);
    printf("\n\n");

    exit(0);
  }

  result = (int*)Malloc(number_to_select * sizeof(int));

  indices_left = (int*)Malloc(number_of_points * sizeof(int));
  for (i = 0; i < number_of_points; i++)
    indices_left[i] = i;

  /* Find the first point: maximum value in a randomly chosen dimension */
  random_dimension_index = randomInt(number_of_dimensions);

  index_of_farthest = 0;
  distance_of_farthest = points[indices_left[index_of_farthest]][random_dimension_index];
  for (i = 1; i < number_of_points; i++) {
    if (points[indices_left[i]][random_dimension_index] > distance_of_farthest) {
      index_of_farthest = i;
      distance_of_farthest = points[indices_left[i]][random_dimension_index];
    }
  }

  number_selected_so_far = 0;
  result[number_selected_so_far] = indices_left[index_of_farthest];
  indices_left[index_of_farthest] = indices_left[number_of_points - number_selected_so_far - 1];
  number_selected_so_far++;

  /* Then select the rest of the solutions: maximum minimum
   * (i.e. nearest-neighbour) distance to so-far selected points */
  nn_distances = (double*)Malloc(number_of_points * sizeof(double));
  for (i = 0; i < number_of_points - number_selected_so_far; i++)
    nn_distances[i] =
        distanceEuclidean(points[indices_left[i]], points[result[number_selected_so_far - 1]], number_of_dimensions);

  while (number_selected_so_far < number_to_select) {
    index_of_farthest = 0;
    distance_of_farthest = nn_distances[0];
    for (i = 1; i < number_of_points - number_selected_so_far; i++) {
      if (nn_distances[i] > distance_of_farthest) {
        index_of_farthest = i;
        distance_of_farthest = nn_distances[i];
      }
    }

    result[number_selected_so_far] = indices_left[index_of_farthest];
    indices_left[index_of_farthest] = indices_left[number_of_points - number_selected_so_far - 1];
    nn_distances[index_of_farthest] = nn_distances[number_of_points - number_selected_so_far - 1];
    number_selected_so_far++;

    for (i = 0; i < number_of_points - number_selected_so_far; i++) {
      value =
          distanceEuclidean(points[indices_left[i]], points[result[number_selected_so_far - 1]], number_of_dimensions);
      if (value < nn_distances[i])
        nn_distances[i] = value;
    }
  }

  free(nn_distances);
  free(indices_left);

  return (result);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Variation -==-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * First estimates the parameters of a normal mixture distribution
 * in the parameter space from the selected solutions. Then copies
 * the best selected solutions. Finally fills up the population,
 * after the variances of the mixture components have been scaled,
 * by drawing new samples from normal mixture distribution and applying
 * AMS to several of these new solutions. Then, the fitness ranks
 * are recomputed. Finally, the distribution multipliers for the
 * mixture components are adapted according to the SDR-AVS mechanism.
 */
void makePopulation(int population_index) {
  current_population_index = population_index;

  estimateParameters(population_index);

  applyDistributionMultipliers(population_index);

  generateAndEvaluateNewSolutionsToFillPopulationAndUpdateElitistArchive(population_index);

  computeRanks(population_index);

  computeObjectiveRanges(population_index);

  adaptObjectiveDiscretization();

  ezilaitiniParametersForSampling(population_index);
}

/**
 * Estimates the parameters of the multivariate normal
 * mixture distribution.
 */
void estimateParameters(int population_index) {
  short *clusters_now_already_registered, *clusters_previous_already_registered;
  int i, j, k, m, q, i_min, j_min, *selection_indices_of_leaders, number_of_dimensions, number_to_select,
      **selection_indices_of_cluster_members_before_registration, *k_means_cluster_sizes,
      **selection_indices_of_cluster_members_k_means, *nearest_neighbour_choice_best,
      number_of_clusters_left_to_register, *sorted, *r_nearest_neighbours_now, *r_nearest_neighbours_previous,
      number_of_clusters_to_register_by_permutation, number_of_cluster_permutations, **all_cluster_permutations;
  double **objective_values_selection_scaled, **objective_values_selection_previous_scaled, distance, distance_smallest,
      distance_largest, **objective_means_scaled_new, **objective_means_scaled_previous, *distances_to_cluster,
      **distance_cluster_i_now_to_cluster_j_previous, **distance_cluster_i_now_to_cluster_j_now,
      **distance_cluster_i_previous_to_cluster_j_previous, epsilon;

  /* Determine the leaders */
  objective_values_selection_scaled = (double**)Malloc(selection_sizes[population_index] * sizeof(double*));
  for (i = 0; i < selection_sizes[population_index]; i++)
    objective_values_selection_scaled[i] = (double*)Malloc(number_of_objectives * sizeof(double));
  for (i = 0; i < selection_sizes[population_index]; i++)
    for (j = 0; j < number_of_objectives; j++)
      objective_values_selection_scaled[i][j] =
          selection[population_index][i]->objective_values[j] / objective_ranges[population_index][j];

  /* Heuristically find k far-apart leaders, taken from an artificial selection */
  int leader_selection_size;

  leader_selection_size = tau * population_sizes[population_index];

  number_of_dimensions = number_of_objectives;
  number_to_select = number_of_mixing_components[population_index];
  selection_indices_of_leaders = greedyScatteredSubsetSelection(
      objective_values_selection_scaled, leader_selection_size, number_of_dimensions, number_to_select);

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    for (j = 0; j < number_of_objectives; j++)
      objective_means_scaled[population_index][i][j] =
          selection[population_index][selection_indices_of_leaders[i]]->objective_values[j] /
          objective_ranges[population_index][j];

  /* Perform k-means clustering with leaders as initial mean guesses */
  objective_means_scaled_new = (double**)Malloc(number_of_mixing_components[population_index] * sizeof(double*));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    objective_means_scaled_new[i] = (double*)Malloc(number_of_objectives * sizeof(double));

  objective_means_scaled_previous = (double**)Malloc(number_of_mixing_components[population_index] * sizeof(double*));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    objective_means_scaled_previous[i] = (double*)Malloc(number_of_objectives * sizeof(double));

  selection_indices_of_cluster_members_k_means =
      (int**)Malloc(number_of_mixing_components[population_index] * sizeof(int*));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    selection_indices_of_cluster_members_k_means[i] = (int*)Malloc(selection_sizes[population_index] * sizeof(int));

  k_means_cluster_sizes = (int*)Malloc(number_of_mixing_components[population_index] * sizeof(int));

  for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++)
    single_objective_clusters[population_index][j] = -1;
  for (j = 0; j < number_of_objectives; j++)
    single_objective_clusters[population_index]
                             [number_of_mixing_components[population_index] - number_of_objectives + j] = j;

  epsilon = 1e+308;
  // BEGIN HACK: This essentially causes the code to skip k-means clustering
  epsilon = 0;
  for (j = 0; j < number_of_mixing_components[population_index]; j++)
    k_means_cluster_sizes[j] = 0;
  // END HACK
  while (epsilon > 1e-10) {
    for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++) {
      k_means_cluster_sizes[j] = 0;
      for (k = 0; k < number_of_objectives; k++)
        objective_means_scaled_new[j][k] = 0.0;
    }

    for (i = 0; i < selection_sizes[population_index]; i++) {
      j_min = -1;
      distance_smallest = -1;
      for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++) {
        distance = distanceEuclidean(objective_values_selection_scaled[i], objective_means_scaled[population_index][j],
                                     number_of_objectives);
        if ((distance_smallest < 0) || (distance < distance_smallest)) {
          j_min = j;
          distance_smallest = distance;
        }
      }
      selection_indices_of_cluster_members_k_means[j_min][k_means_cluster_sizes[j_min]] = i;
      for (k = 0; k < number_of_objectives; k++)
        objective_means_scaled_new[j_min][k] += objective_values_selection_scaled[i][k];
      k_means_cluster_sizes[j_min]++;
    }

    for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++)
      for (k = 0; k < number_of_objectives; k++)
        objective_means_scaled_new[j][k] /= (double)k_means_cluster_sizes[j];

    epsilon = 0;
    for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++) {
      epsilon += distanceEuclidean(objective_means_scaled[population_index][j], objective_means_scaled_new[j],
                                   number_of_objectives);
      for (k = 0; k < number_of_objectives; k++)
        objective_means_scaled[population_index][j][k] = objective_means_scaled_new[j][k];
    }
  }

  /* Do leader-based distance assignment */
  distances_to_cluster = (double*)Malloc(selection_sizes[population_index] * sizeof(double));
  for (i = 0; i < number_of_mixing_components[population_index] - number_of_objectives; i++) {
    for (j = 0; j < selection_sizes[population_index]; j++)
      distances_to_cluster[j] = distanceEuclidean(objective_values_selection_scaled[j],
                                                  objective_means_scaled[population_index][i], number_of_objectives);
    for (j = leader_selection_size; j < selection_sizes[population_index]; j++)
      distances_to_cluster[j] = 1e+308;  // HACK

    if (selection_indices_of_cluster_members_previous[population_index][i] != NULL)
      free(selection_indices_of_cluster_members_previous[population_index][i]);
    selection_indices_of_cluster_members_previous[population_index][i] =
        selection_indices_of_cluster_members[population_index][i];
    selection_indices_of_cluster_members[population_index][i] =
        mergeSort(distances_to_cluster, selection_sizes[population_index]);
  }

  // For k-th objective, create a cluster consisting of only the best solutions in that objective (from the overall
  // selection)
  for (j = number_of_mixing_components[population_index] - number_of_objectives;
       j < number_of_mixing_components[population_index]; j++) {
    double *individual_objectives, worst;

    individual_objectives = (double*)Malloc(selection_sizes[population_index] * sizeof(double));

    worst = -1e+308;
    for (i = 0; i < selection_sizes[population_index]; i++) {
      individual_objectives[i] =
          selection[population_index][i]
              ->objective_values[j - (number_of_mixing_components[population_index] - number_of_objectives)];
      if (individual_objectives[i] > worst)
        worst = individual_objectives[i];
    }
    for (i = 0; i < selection_sizes[population_index]; i++) {
      if (selection[population_index][i]->constraint_value != 0)
        individual_objectives[i] = worst + selection[population_index][i]->constraint_value;
    }

    if (selection_indices_of_cluster_members_previous[population_index][j] != NULL)
      free(selection_indices_of_cluster_members_previous[population_index][j]);
    selection_indices_of_cluster_members_previous[population_index][j] =
        selection_indices_of_cluster_members[population_index][j];
    selection_indices_of_cluster_members[population_index][j] =
        mergeSort(individual_objectives, selection_sizes[population_index]);
    free(individual_objectives);
  }

  /* Re-assign cluster indices to achieve cluster registration,
   * i.e. make cluster i in this generation to be the cluster that is
   * closest to cluster i of the previous generation. The
   * algorithm first computes all distances between clusters in
   * the current generation and the previous generation. It also
   * computes all distances between the clusters in the current
   * generation and all distances between the clusters in the
   * previous generation. Then it determines the two clusters
   * that are the farthest apart. It randomly takes one of
   * these two far-apart clusters and its r nearest neighbours.
   * It also finds the closest cluster among those of the previous
   * generation and its r nearest neighbours. All permutations
   * are then considered to register these two sets. Subset
   * registration continues in this fashion until all clusters
   * are registered. */
  if (number_of_generations[population_index] > 0) {
    number_of_nearest_neighbours_in_registration = 7;

    objective_values_selection_previous_scaled = (double**)Malloc(selection_sizes[population_index] * sizeof(double*));
    for (i = 0; i < selection_sizes[population_index]; i++)
      objective_values_selection_previous_scaled[i] = (double*)Malloc(number_of_objectives * sizeof(double));

    for (i = 0; i < selection_sizes[population_index]; i++)
      for (j = 0; j < number_of_objectives; j++)
        objective_values_selection_previous_scaled[i][j] =
            objective_values_selection_previous[population_index][i][j] / objective_ranges[population_index][j];

    selection_indices_of_cluster_members_before_registration =
        (int**)Malloc(number_of_mixing_components[population_index] * sizeof(int*));
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      selection_indices_of_cluster_members_before_registration[i] =
          selection_indices_of_cluster_members[population_index][i];

    distance_cluster_i_now_to_cluster_j_previous =
        (double**)Malloc(number_of_mixing_components[population_index] * sizeof(double*));
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      distance_cluster_i_now_to_cluster_j_previous[i] =
          (double*)Malloc(number_of_mixing_components[population_index] * sizeof(double));

    /* Compute distances between clusters */
    ////// START DISTANCE COMPUTATION
    /* OLD: distance between clusters is the smallest distance between pairs of points.
     */
    for (i = 0; i < number_of_mixing_components[population_index]; i++) {
      for (j = 0; j < number_of_mixing_components[population_index]; j++) {
        distance_cluster_i_now_to_cluster_j_previous[i][j] = 0;
        for (k = 0; k < cluster_sizes[population_index]; k++) {
          distance_smallest = -1;
          for (q = 0; q < cluster_sizes[population_index]; q++) {
            distance = distanceEuclidean(
                objective_values_selection_scaled[selection_indices_of_cluster_members_before_registration[i][k]],
                objective_values_selection_previous_scaled
                    [selection_indices_of_cluster_members_previous[population_index][j][q]],
                number_of_objectives);
            if ((distance_smallest < 0) || (distance < distance_smallest))
              distance_smallest = distance;
          }
          distance_cluster_i_now_to_cluster_j_previous[i][j] += distance_smallest;
        }
      }
    }

    distance_cluster_i_now_to_cluster_j_now =
        (double**)Malloc(number_of_mixing_components[population_index] * sizeof(double*));
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      distance_cluster_i_now_to_cluster_j_now[i] =
          (double*)Malloc(number_of_mixing_components[population_index] * sizeof(double));

    for (i = 0; i < number_of_mixing_components[population_index]; i++) {
      for (j = 0; j < number_of_mixing_components[population_index]; j++) {
        distance_cluster_i_now_to_cluster_j_now[i][j] = 0;
        if (i != j) {
          for (k = 0; k < cluster_sizes[population_index]; k++) {
            distance_smallest = -1;
            for (q = 0; q < cluster_sizes[population_index]; q++) {
              distance = distanceEuclidean(
                  objective_values_selection_scaled[selection_indices_of_cluster_members_before_registration[i][k]],
                  objective_values_selection_scaled[selection_indices_of_cluster_members_before_registration[j][q]],
                  number_of_objectives);
              if ((distance_smallest < 0) || (distance < distance_smallest))
                distance_smallest = distance;
            }
            distance_cluster_i_now_to_cluster_j_now[i][j] += distance_smallest;
          }
        }
      }
    }

    distance_cluster_i_previous_to_cluster_j_previous =
        (double**)Malloc(number_of_mixing_components[population_index] * sizeof(double*));
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      distance_cluster_i_previous_to_cluster_j_previous[i] =
          (double*)Malloc(number_of_mixing_components[population_index] * sizeof(double));

    for (i = 0; i < number_of_mixing_components[population_index]; i++) {
      for (j = 0; j < number_of_mixing_components[population_index]; j++) {
        distance_cluster_i_previous_to_cluster_j_previous[i][j] = 0;
        if (i != j) {
          for (k = 0; k < cluster_sizes[population_index]; k++) {
            distance_smallest = -1;
            for (q = 0; q < cluster_sizes[population_index]; q++) {
              distance = distanceEuclidean(objective_values_selection_previous_scaled
                                               [selection_indices_of_cluster_members_previous[population_index][i][k]],
                                           objective_values_selection_previous_scaled
                                               [selection_indices_of_cluster_members_previous[population_index][j][q]],
                                           number_of_objectives);
              if ((distance_smallest < 0) || (distance < distance_smallest))
                distance_smallest = distance;
            }
            distance_cluster_i_previous_to_cluster_j_previous[i][j] += distance_smallest;
          }
        }
      }
    }
    /* NEW: distance between clusters is the distance between the cluster means in scaled-objective space.
        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled[population_index][j][k] = 0.0;

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            for( q = 0; q < cluster_sizes[population_index]; q++ )
              objective_means_scaled[population_index][j][k] +=
       objective_values_selection_scaled[selection_indices_of_cluster_members[population_index][j][q]][k];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled[population_index][j][k] /= (double) cluster_sizes[population_index];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled_previous[j][k] = 0.0;

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            for( q = 0; q < cluster_sizes[population_index]; q++ )
              objective_means_scaled_previous[j][k] +=
       objective_values_selection_previous_scaled[selection_indices_of_cluster_members_previous[population_index][j][q]][k];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled_previous[j][k] /= (double) cluster_sizes[population_index];

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = i; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_now_to_cluster_j_previous[i][j] = distanceEuclidean(
       objective_means_scaled[population_index][i], objective_means_scaled_previous[j], number_of_objectives );
            distance_cluster_i_now_to_cluster_j_previous[j][i] = distance_cluster_i_now_to_cluster_j_previous[i][j];
          }
        }

        distance_cluster_i_now_to_cluster_j_now = (double **) Malloc(
       number_of_mixing_components[population_index]*sizeof( double * ) ); for( i = 0; i <
       number_of_mixing_components[population_index]; i++ ) distance_cluster_i_now_to_cluster_j_now[i] = (double *)
       Malloc( number_of_mixing_components[population_index]*sizeof( double ) );

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = i; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_now_to_cluster_j_now[i][j] = 0;
            if( i != j )
            {
              distance_cluster_i_now_to_cluster_j_now[i][j] = distanceEuclidean(
       objective_means_scaled[population_index][i], objective_means_scaled[population_index][j], number_of_objectives );
              distance_cluster_i_now_to_cluster_j_now[j][i] = distance_cluster_i_now_to_cluster_j_now[i][j];
            }
          }
        }

        distance_cluster_i_previous_to_cluster_j_previous = (double **) Malloc(
       number_of_mixing_components[population_index]*sizeof( double * ) ); for( i = 0; i < number_of_mixing_componen
              for( k = 0; k < number_of_parameters; k++ )ts; i++ )
          distance_cluster_i_previous_to_cluster_j_previous[i] = (double *) Malloc(
       number_of_mixing_components[population_index]*sizeof( double ) );

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_previous_to_cluster_j_previous[i][j] = 0;
            if( i != j )
            {
              distance_cluster_i_previous_to_cluster_j_previous[i][j] = distanceEuclidean(
       objective_means_scaled_previous[i], objective_means_scaled_previous[j], number_of_objectives );
              distance_cluster_i_previou- number_of_objectivess_to_cluster_j_previous[j][i] =
       distance_cluster_i_previous_to_cluster_j_previous[i][j];
            }
          }
        }
    */
    /* NEWNEW: distance between clusters is RANDOM
        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled[population_index][j][k] = 0.0;

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            for( q = 0; q < cluster_sizes[population_index]; q++ )
              objective_means_scaled[population_index][j][k] +=
       objective_values_selection_scaled[selection_indices_of_cluster_members[population_index][j][q]][k];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled[population_index][j][k] /= (double) cluster_sizes[population_index];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled_previous[j][k] = 0.0;

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            for( q = 0; q < cluster_sizes[population_index]; q++ )
              objective_means_scaled_previous[j][k] +=
       objective_values_selection_previous_scaled[selection_indices_of_cluster_members_previous[population_index][j][q]][k];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled_previous[j][k] /= (double) cluster_sizes[population_index];

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = i; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_now_to_cluster_j_previous[i][j] = randomRealUniform01();
            distance_cluster_i_now_to_cluster_j_previous[j][i] = distance_cluster_i_now_to_cluster_j_previous[i][j];
          }
        }

        distance_cluster_i_now_to_cluster_j_now = (double **) Malloc(
       number_of_mixing_components[population_index]*sizeof( double * ) ); for( i = 0; i <
       number_of_mixing_components[population_index]; i++ ) distance_cluster_i_now_to_cluster_j_now[i] = (double *)
       Malloc( number_of_mixing_components[population_index]*sizeof( double ) );

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = i; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_now_to_cluster_j_now[i][j] = 0;
            if( i != j )
            {
              distance_cluster_i_now_to_cluster_j_now[i][j] = randomRealUniform01();
              distance_cluster_i_now_to_cluster_j_now[j][i] = distance_cluster_i_now_to_cluster_j_now[i][j];
            }
          }
        }

        distance_cluster_i_previous_to_cluster_j_previous = (double **) Malloc(
       number_of_mixing_components[population_index]*sizeof( double * ) ); for( i = 0; i <
       number_of_mixing_components[population_index]; i++ ) distance_cluster_i_previous_to_cluster_j_previous[i] =
       (double *) Malloc( number_of_mixing_components[population_index]*sizeof( double ) );

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_previous_to_cluster_j_previous[i][j] = 0;
            if( i != j )
            {
              distance_cluster_i_previous_to_cluster_j_previous[i][j] = randomRealUniform01();
              distance_cluster_i_previous_to_cluster_j_previous[j][i] =
       distance_cluster_i_previous_to_cluster_j_previous[i][j];
            }
          }
        }
    */
    ////// END DISTANCE COMPUTATION

    clusters_now_already_registered = (short*)Malloc(number_of_mixing_components[population_index] * sizeof(short));
    clusters_previous_already_registered =
        (short*)Malloc(number_of_mixing_components[population_index] * sizeof(short));
    for (i = 0; i < number_of_mixing_components[population_index]; i++) {
      clusters_now_already_registered[i] = 0;
      clusters_previous_already_registered[i] = 0;
    }

    r_nearest_neighbours_now = (int*)Malloc((number_of_nearest_neighbours_in_registration + 1) * sizeof(int));
    r_nearest_neighbours_previous = (int*)Malloc((number_of_nearest_neighbours_in_registration + 1) * sizeof(int));
    nearest_neighbour_choice_best = (int*)Malloc((number_of_nearest_neighbours_in_registration + 1) * sizeof(int));

    number_of_clusters_left_to_register = number_of_mixing_components[population_index];
    while (number_of_clusters_left_to_register > 0) {
      /* Find the two clusters in the current generation that are farthest apart and haven't been registered yet */
      i_min = -1;
      j_min = -1;
      distance_largest = -1;
      for (i = 0; i < number_of_mixing_components[population_index]; i++) {
        if (clusters_now_already_registered[i] == 0) {
          for (j = 0; j < number_of_mixing_components[population_index]; j++) {
            if ((i != j) && (clusters_now_already_registered[j] == 0)) {
              distance = distance_cluster_i_now_to_cluster_j_now[i][j];
              if ((distance_largest < 0) || (distance > distance_largest)) {
                distance_largest = distance;
                i_min = i;
                j_min = j;
              }
            }
          }
        }
      }

      if (i_min == -1) {
        for (i = 0; i < number_of_mixing_components[population_index]; i++)
          if (clusters_now_already_registered[i] == 0) {
            i_min = i;
            break;
          }
      }

      /* Find the r nearest clusters of one of the two far-apart clusters that haven't been registered yet */
      sorted = mergeSort(distance_cluster_i_now_to_cluster_j_now[i_min], number_of_mixing_components[population_index]);
      j = 0;
      for (i = 0; i < number_of_mixing_components[population_index]; i++) {
        if (clusters_now_already_registered[sorted[i]] == 0) {
          r_nearest_neighbours_now[j] = sorted[i];
          clusters_now_already_registered[sorted[i]] = 1;
          j++;
        }
        if (j == number_of_nearest_neighbours_in_registration && number_of_clusters_left_to_register - j != 1)
          break;
      }
      number_of_clusters_to_register_by_permutation = j;
      free(sorted);

      /* Find the closest cluster from the previous generation */
      j_min = -1;
      distance_smallest = -1;
      for (j = 0; j < number_of_mixing_components[population_index]; j++) {
        if (clusters_previous_already_registered[j] == 0) {
          distance = distance_cluster_i_now_to_cluster_j_previous[i_min][j];
          if ((distance_smallest < 0) || (distance < distance_smallest)) {
            distance_smallest = distance;
            j_min = j;
          }
        }
      }

      /* Find the r nearest clusters of one of the the closest cluster from the previous generation */
      sorted = mergeSort(distance_cluster_i_previous_to_cluster_j_previous[j_min],
                         number_of_mixing_components[population_index]);
      j = 0;
      for (i = 0; i < number_of_mixing_components[population_index]; i++) {
        if (clusters_previous_already_registered[sorted[i]] == 0) {
          r_nearest_neighbours_previous[j] = sorted[i];
          clusters_previous_already_registered[sorted[i]] = 1;
          j++;
        }
        if (j == number_of_clusters_to_register_by_permutation)
          break;
      }
      free(sorted);

      /* Register the r selected clusters from the current and the previous generation */
      all_cluster_permutations =
          allPermutations(number_of_clusters_to_register_by_permutation, &number_of_cluster_permutations);
      distance_smallest = -1;
      for (i = 0; i < number_of_cluster_permutations; i++) {
        distance = 0;
        for (j = 0; j < number_of_clusters_to_register_by_permutation; j++)
          distance += distance_cluster_i_now_to_cluster_j_previous
              [r_nearest_neighbours_now[j]][r_nearest_neighbours_previous[all_cluster_permutations[i][j]]];
        if ((distance_smallest < 0) || (distance < distance_smallest)) {
          distance_smallest = distance;
          for (j = 0; j < number_of_clusters_to_register_by_permutation; j++)
            nearest_neighbour_choice_best[j] = r_nearest_neighbours_previous[all_cluster_permutations[i][j]];
        }
      }
      for (i = 0; i < number_of_cluster_permutations; i++)
        free(all_cluster_permutations[i]);
      free(all_cluster_permutations);

      for (i = 0; i < number_of_clusters_to_register_by_permutation; i++) {
        selection_indices_of_cluster_members[population_index][nearest_neighbour_choice_best[i]] =
            selection_indices_of_cluster_members_before_registration[r_nearest_neighbours_now[i]];
        if (r_nearest_neighbours_now[i] >= number_of_mixing_components[population_index] - number_of_objectives) {
          single_objective_clusters[population_index][nearest_neighbour_choice_best[i]] =
              r_nearest_neighbours_now[i] - (number_of_mixing_components[population_index] - number_of_objectives);
          single_objective_clusters[population_index][r_nearest_neighbours_now[i]] = -1;
        }
      }

      number_of_clusters_left_to_register -= number_of_clusters_to_register_by_permutation;
    }

    free(nearest_neighbour_choice_best);
    free(r_nearest_neighbours_previous);
    free(r_nearest_neighbours_now);
    free(clusters_now_already_registered);
    free(clusters_previous_already_registered);
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      free(distance_cluster_i_previous_to_cluster_j_previous[i]);
    free(distance_cluster_i_previous_to_cluster_j_previous);
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      free(distance_cluster_i_now_to_cluster_j_now[i]);
    free(distance_cluster_i_now_to_cluster_j_now);
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      free(distance_cluster_i_now_to_cluster_j_previous[i]);
    free(distance_cluster_i_now_to_cluster_j_previous);
    free(selection_indices_of_cluster_members_before_registration);
    for (i = 0; i < selection_sizes[population_index]; i++)
      free(objective_values_selection_previous_scaled[i]);
    free(objective_values_selection_previous_scaled);
  }

  // Compute objective means
  for (j = 0; j < number_of_mixing_components[population_index]; j++)
    for (k = 0; k < number_of_objectives; k++)
      objective_means_scaled[population_index][j][k] = 0.0;

  for (j = 0; j < number_of_mixing_components[population_index]; j++)
    for (k = 0; k < number_of_objectives; k++)
      for (q = 0; q < cluster_sizes[population_index]; q++)
        objective_means_scaled[population_index][j][k] +=
            objective_values_selection_scaled[selection_indices_of_cluster_members[population_index][j][q]][k];

  for (j = 0; j < number_of_mixing_components[population_index]; j++)
    for (k = 0; k < number_of_objectives; k++)
      objective_means_scaled[population_index][j][k] /= (double)cluster_sizes[population_index];

  int** full_rankings = (int**)Malloc(number_of_mixing_components[population_index] * sizeof(int*));
  double* distances = (double*)Malloc(selection_sizes[population_index] * sizeof(double));
  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    for (j = 0; j < selection_sizes[population_index]; j++)
      distances[j] = distanceEuclidean(objective_values_selection_scaled[j],
                                       objective_means_scaled[population_index][i], number_of_objectives);
    full_rankings[i] = mergeSort(distances, selection_sizes[population_index]);
  }

  // Assign exactly 'cluster_size' individuals of the population to each cluster
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    num_individuals_in_cluster[population_index][i] = 0;
  for (i = 0; i < population_sizes[population_index]; i++)
    cluster_index_for_population[population_index][i] = -1;

  for (j = 0; j < cluster_sizes[population_index]; j++) {
    for (i = number_of_mixing_components[population_index] - 1; i >= 0; i--) {
      int inc = 0;
      int individual_index =
          selection_indices[population_index][selection_indices_of_cluster_members[population_index][i][j]];
      while (cluster_index_for_population[population_index][individual_index] != -1) {
        individual_index = selection_indices[population_index][full_rankings[i][j + inc]];
        inc++;
      }
      cluster_index_for_population[population_index][individual_index] = i;
      num_individuals_in_cluster[population_index][i]++;
    }
  }
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(full_rankings[i]);
  free(full_rankings);
  free(distances);

  double* objective_values_scaled = (double*)Malloc(number_of_objectives * sizeof(double));
  int index_smallest;
  for (i = 0; i < population_sizes[population_index]; i++) {
    if (cluster_index_for_population[population_index][i] != -1)
      continue;

    for (j = 0; j < number_of_objectives; j++)
      objective_values_scaled[j] =
          populations[population_index][i]->objective_values[j] / objective_ranges[population_index][j];

    distance_smallest = -1;
    index_smallest = -1;
    for (j = 0; j < number_of_mixing_components[population_index]; j++) {
      distance =
          distanceEuclidean(objective_values_scaled, objective_means_scaled[population_index][j], number_of_objectives);
      if ((distance_smallest < 0) || (distance < distance_smallest)) {
        index_smallest = j;
        distance_smallest = distance;
      }
    }
    cluster_index_for_population[population_index][i] = index_smallest;
    num_individuals_in_cluster[population_index][index_smallest]++;
  }
  free(objective_values_scaled);

  /* Elitism, must be done here, before possibly changing the focus of each cluster to an elitist solution */
  copyBestSolutionsToPopulation(population_index, objective_values_selection_scaled);

  /* Estimate the parameters */
  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    /* Means */
    if (number_of_generations[population_index] > 0) {
      for (j = 0; j < number_of_parameters; j++)
        mean_vectors_previous[population_index][i][j] = mean_vectors[population_index][i][j];
    }

    for (j = 0; j < number_of_parameters; j++) {
      mean_vectors[population_index][i][j] = 0.0;

      for (k = 0; k < cluster_sizes[population_index]; k++)
        mean_vectors[population_index][i][j] +=
            selection[population_index][selection_indices_of_cluster_members[population_index][i][k]]->parameters[j];

      mean_vectors[population_index][i][j] /= (double)cluster_sizes[population_index];
    }
  }

  if (learn_linkage_tree) {
    for (i = 0; i < number_of_mixing_components[population_index]; i++) {
      estimateFullCovarianceMatrixML(population_index, i);

      linkage_model[population_index][i] = learnLinkageTreeRVGOMEA(population_index, i);

      for (j = 0; j < number_of_parameters; j++)
        free(full_covariance_matrix[population_index][i][j]);
      free(full_covariance_matrix[population_index][i]);
    }

    initializeCovarianceMatrices(population_index);

    if (number_of_generations[population_index] == 0)
      initializeDistributionMultipliers(population_index);
  }

  int vara, varb, cluster_index;
  double cov;
  for (cluster_index = 0; cluster_index < number_of_mixing_components[population_index]; cluster_index++) {
    /* First do the maximum-likelihood estimate from data */
    for (i = 0; i < linkage_model[population_index][cluster_index]->length; i++) {
      for (j = 0; j < linkage_model[population_index][cluster_index]->set_length[i]; j++) {
        vara = linkage_model[population_index][cluster_index]->sets[i][j];
        for (k = j; k < linkage_model[population_index][cluster_index]->set_length[i]; k++) {
          varb = linkage_model[population_index][cluster_index]->sets[i][k];
          cov = 0.0;

          for (m = 0; m < cluster_sizes[population_index]; m++)
            cov +=
                (selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][m]]
                     ->parameters[vara] -
                 mean_vectors[population_index][cluster_index][vara]) *
                (selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][m]]
                     ->parameters[varb] -
                 mean_vectors[population_index][cluster_index][varb]);

          cov /= (double)cluster_sizes[population_index];
          decomposed_covariance_matrices[population_index][cluster_index][i][j][k] = cov;
          decomposed_covariance_matrices[population_index][cluster_index][i][k][j] = cov;
        }
      }
    }
  }

  free(distances_to_cluster);
  free(k_means_cluster_sizes);
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(selection_indices_of_cluster_members_k_means[i]);
  free(selection_indices_of_cluster_members_k_means);
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(objective_means_scaled_new[i]);
  free(objective_means_scaled_new);
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(objective_means_scaled_previous[i]);
  free(objective_means_scaled_previous);
  for (i = 0; i < selection_sizes[population_index]; i++)
    free(objective_values_selection_scaled[i]);
  free(objective_values_selection_scaled);
  free(selection_indices_of_leaders);
}

/**
 * Elitism: copies at most 1/k*tau*n solutions per cluster
 * from the elitist archive.
 */
void copyBestSolutionsToPopulation(int population_index, double** objective_values_selection_scaled) {
  int i, j, j_min, k, index, **elitist_archive_indices_per_cluster, so_index,
      *number_of_elitist_archive_indices_per_cluster, max, *diverse_indices, skipped;
  double distance, distance_smallest, *objective_values_scaled, **points;

  number_of_elitist_archive_indices_per_cluster =
      (int*)Malloc(number_of_mixing_components[population_index] * sizeof(int));
  elitist_archive_indices_per_cluster = (int**)Malloc(number_of_mixing_components[population_index] * sizeof(int*));
  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    number_of_elitist_archive_indices_per_cluster[i] = 0;
    number_of_elitist_solutions_copied[population_index][i] = 0;
    elitist_archive_indices_per_cluster[i] = (int*)Malloc(elitist_archive_size * sizeof(int));
  }
  objective_values_scaled = (double*)Malloc(number_of_objectives * sizeof(double));

  for (i = 0; i < elitist_archive_size; i++) {
    if (elitist_archive_indices_inactive[i])
      continue;
    for (j = 0; j < number_of_objectives; j++)
      objective_values_scaled[j] = elitist_archive[i]->objective_values[j] / objective_ranges[population_index][j];
    j_min = -1;
    distance_smallest = -1;
    for (j = 0; j < number_of_mixing_components[population_index]; j++) {
      distance =
          distanceEuclidean(objective_values_scaled, objective_means_scaled[population_index][j], number_of_objectives);
      if ((distance_smallest < 0) || (distance < distance_smallest)) {
        j_min = j;
        distance_smallest = distance;
      }
    }

    elitist_archive_indices_per_cluster[j_min][number_of_elitist_archive_indices_per_cluster[j_min]] = i;
    number_of_elitist_archive_indices_per_cluster[j_min]++;
  }

  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    max = (int)(tau * num_individuals_in_cluster[population_index][i]);
    skipped = 0;
    if (number_of_elitist_archive_indices_per_cluster[i] <= max) {
      for (j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++) {
        index = sorted_ranks[population_index][population_sizes[population_index] - 1 - skipped];  // BLA
        while (cluster_index_for_population[population_index][index] != i &&
               (population_sizes[population_index] - 1 - skipped) > 0)
          index = sorted_ranks[population_index][population_sizes[population_index] - 1 - (++skipped)];
        if (cluster_index_for_population[population_index][index] != i)
          break;
        so_index = single_objective_clusters[population_index][i];
        if (so_index != -1 &&
            populations[population_index][index]->objective_values[so_index] <
                elitist_archive[elitist_archive_indices_per_cluster[i][j]]->objective_values[so_index])
          continue;
        copyIndividual(elitist_archive[elitist_archive_indices_per_cluster[i][j]],
                       populations[population_index][index]);
        populations[population_index][index]->NIS = 0;
        skipped++;
      }
      number_of_elitist_solutions_copied[population_index][i] = j;
    } else {
      points = (double**)Malloc(number_of_elitist_archive_indices_per_cluster[i] * sizeof(double*));
      for (j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++)
        points[j] = (double*)Malloc(number_of_objectives * sizeof(double));
      for (j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++) {
        for (k = 0; k < number_of_objectives; k++)
          points[j][k] = elitist_archive[elitist_archive_indices_per_cluster[i][j]]->objective_values[k] /
                         objective_ranges[population_index][k];
      }
      diverse_indices = greedyScatteredSubsetSelection(points, number_of_elitist_archive_indices_per_cluster[i],
                                                       number_of_objectives, max);
      for (j = 0; j < max; j++) {
        index = sorted_ranks[population_index][population_sizes[population_index] - 1 - skipped];  // BLA
        while (cluster_index_for_population[population_index][index] != i &&
               (population_sizes[population_index] - 1 - skipped) > 0)
          index = sorted_ranks[population_index][population_sizes[population_index] - 1 - (++skipped)];
        if (cluster_index_for_population[population_index][index] != i)
          break;
        so_index = single_objective_clusters[population_index][i];
        if (so_index != -1 &&
            populations[population_index][index]->objective_values[so_index] <
                elitist_archive[elitist_archive_indices_per_cluster[i][j]]->objective_values[so_index])
          continue;
        copyIndividual(elitist_archive[elitist_archive_indices_per_cluster[i][diverse_indices[j]]],
                       populations[population_index][index]);
        populations[population_index][index]->NIS = 0;
        skipped++;
      }
      number_of_elitist_solutions_copied[population_index][i] = j;
      free(diverse_indices);
      for (j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++)
        free(points[j]);
      free(points);
    }
  }

  free(objective_values_scaled);
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(elitist_archive_indices_per_cluster[i]);
  free(elitist_archive_indices_per_cluster);
  free(number_of_elitist_archive_indices_per_cluster);
}

/**
 * Initializes the FOS
 */
void initializeFOS(int population_index, int cluster_index) {
  int i, j;
  FILE* file;
  FOS* new_FOS;

  // fflush(stdout);
  // file = fopen("FOS.in", "r");
  // if (file != NULL) {
  //   if (population_index == 0 && cluster_index == 0)
  //     new_FOS = readFOSFromFile(file);
  //   else
  //     new_FOS = copyFOS(linkage_model[0][0]);
  if (global_static_fos != NULL) {
    const goblin::FOS& fos = global_static_fos->get();

    new_FOS = (FOS*)Malloc(sizeof(FOS));
    new_FOS->length = fos.size();
    new_FOS->set_length = (int*)Malloc(new_FOS->length * sizeof(int));
    new_FOS->sets = (int**)Malloc(new_FOS->length * sizeof(int*));
    for (i = 0; i < new_FOS->length; i++) {
      new_FOS->set_length[i] = fos[i].continuous.size();
      new_FOS->sets[i] = (int*)Malloc(new_FOS->set_length[i] * sizeof(int));
      for (j = 0; j < new_FOS->set_length[i]; j++)
        new_FOS->sets[i][j] = fos[i].continuous[j];
    }
  } else {
    if (static_linkage_tree) {
      if (population_index == 0 && cluster_index == 0) {
        new_FOS = learnLinkageTree(NULL);
      } else
        new_FOS = copyFOS(linkage_model[0][0]);
    } else {
      new_FOS = (FOS*)Malloc(sizeof(FOS));
      new_FOS->length = (int)((number_of_parameters + FOS_element_size - 1) / FOS_element_size);
      new_FOS->sets = (int**)Malloc(new_FOS->length * sizeof(int*));
      new_FOS->set_length = (int*)Malloc(new_FOS->length * sizeof(int));
      for (i = 0; i < new_FOS->length; i++) {
        new_FOS->sets[i] = (int*)Malloc(FOS_element_size * sizeof(int));
        new_FOS->set_length[i] = 0;
      }

      for (i = 0; i < number_of_parameters; i++) {
        new_FOS->sets[i / FOS_element_size][i % FOS_element_size] = i;
        new_FOS->set_length[i / FOS_element_size]++;
      }
    }
  }
  linkage_model[population_index][cluster_index] = new_FOS;
}

FOS* learnLinkageTreeRVGOMEA(int population_index, int cluster_index) {
  int i;
  FOS* new_FOS;

  new_FOS = learnLinkageTree(full_covariance_matrix[population_index][cluster_index]);
  if (learn_linkage_tree && number_of_generations[population_index] > 0)
    inheritDistributionMultipliers(new_FOS, linkage_model[population_index][cluster_index],
                                   distribution_multipliers[population_index][cluster_index]);

  if (learn_linkage_tree && number_of_generations[population_index] > 0) {
    for (i = 0; i < linkage_model[population_index][cluster_index]->length; i++)
      free(linkage_model[population_index][cluster_index]->sets[i]);
    free(linkage_model[population_index][cluster_index]->sets);
    free(linkage_model[population_index][cluster_index]->set_length);
    free(linkage_model[population_index][cluster_index]);
  }
  return (new_FOS);
}

void inheritDistributionMultipliers(FOS* new_FOS, FOS* prev_FOS, double* multipliers) {
  int i, *permutation;
  double* multipliers_copy;

  multipliers_copy = (double*)Malloc(new_FOS->length * sizeof(double));
  for (i = 0; i < new_FOS->length; i++)
    multipliers_copy[i] = multipliers[i];

  permutation = matchFOSElements(new_FOS, prev_FOS);

  for (i = 0; i < new_FOS->length; i++)
    multipliers[permutation[i]] = multipliers_copy[i];

  free(multipliers_copy);
  free(permutation);
}

void estimateFullCovarianceMatrixML(int population_index, int cluster_index) {
  int i, j, k, q;
  double cov;

  i = cluster_index;
  full_covariance_matrix[population_index][i] = (double**)Malloc(number_of_parameters * sizeof(double*));
  for (k = 0; k < number_of_parameters; k++)
    full_covariance_matrix[population_index][i][k] = (double*)Malloc(number_of_parameters * sizeof(double));

  /* Covariance matrices */
  for (j = 0; j < number_of_parameters; j++) {
    for (q = j; q < number_of_parameters; q++) {
      cov = 0.0;
      for (k = 0; k < cluster_sizes[population_index]; k++)
        cov +=
            (selection[population_index][selection_indices_of_cluster_members[population_index][i][k]]->parameters[j] -
             mean_vectors[population_index][i][j]) *
            (selection[population_index][selection_indices_of_cluster_members[population_index][i][k]]->parameters[q] -
             mean_vectors[population_index][i][q]);
      cov /= (double)cluster_sizes[population_index];

      full_covariance_matrix[population_index][i][j][q] = cov;
      full_covariance_matrix[population_index][i][q][j] = cov;
    }
  }
}

void evaluateCompletePopulation(int population_index) {
  int i;
  for (i = 0; i < population_sizes[population_index]; i++) {
    installedProblemEvaluation(problem_index, populations[population_index][i], number_of_parameters, NULL, NULL, 0, 0);
    if (global_terminate_immediately) {
      return;
    }
  }
}

/**
 * Applies the distribution multipliers.
 */
void applyDistributionMultipliers(int population_index) {
  int i, j, k, m;

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
      for (k = 0; k < linkage_model[population_index][i]->set_length[j]; k++)
        for (m = 0; m < linkage_model[population_index][i]->set_length[j]; m++)
          decomposed_covariance_matrices[population_index][i][j][k][m] *=
              distribution_multipliers[population_index][i][j];
}

/**
 * Generates new solutions by sampling the mixture distribution.
 */
void generateAndEvaluateNewSolutionsToFillPopulationAndUpdateElitistArchive(int population_index) {
  short cluster_failure, all_multipliers_leq_one, *generational_improvement, any_improvement, *is_improved_by_AMS;
  int i, j, k, m, oj, c, *order;

  if (!black_box_evaluations && (number_of_generations[population_index] + 1) % 50 == 0)
    evaluateCompletePopulation(population_index);

  if (global_terminate_immediately) {
    return;
  }

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    computeParametersForSampling(population_index, i);

  generational_improvement = (short*)Malloc(population_sizes[population_index] * sizeof(short));

  for (i = 0; i < population_sizes[population_index]; i++)
    generational_improvement[i] = 0;

  for (k = 0; k < number_of_mixing_components[population_index]; k++) {
    order = randomPermutation(linkage_model[population_index][k]->length);
    for (m = 0; m < linkage_model[population_index][k]->length; m++) {
      samples_current_cluster = 0;
      oj = order[m];
      samples_drawn_from_normal[population_index][k][oj] = 0;
      out_of_bounds_draws[population_index][k][oj] = 0;

      for (i = 0; i < population_sizes[population_index]; i++) {
        if (cluster_index_for_population[population_index][i] != k)
          continue;
        if (generateNewSolutionFromFOSElement(population_index, k, oj, i)) {
          generational_improvement[i] = 1;

          if (global_terminate_immediately) {
            free(order);
            free(generational_improvement);
            return;
          }
        }
        samples_current_cluster++;
      }

      adaptDistributionMultipliers(population_index, k, oj);
      for (i = 0; i < population_sizes[population_index]; i++)
        if (cluster_index_for_population[population_index][i] == k && generational_improvement[i])
          updateElitistArchive(populations[population_index][i]);
    }
    free(order);

    c = 0;
    if (number_of_generations[population_index] > 0) {
      is_improved_by_AMS = (short*)Malloc(population_sizes[population_index] * sizeof(short));
      for (i = 0; i < population_sizes[population_index]; i++)
        is_improved_by_AMS[i] = 0;
      for (i = 0; i < population_sizes[population_index]; i++) {
        if (cluster_index_for_population[population_index][i] != k)
          continue;
        is_improved_by_AMS[i] = applyAMS(population_index, i, k);
        generational_improvement[i] |= is_improved_by_AMS[i];

        c++;
        if (c >= 0.5 * tau * num_individuals_in_cluster[population_index][k])
          break;

        if (global_terminate_immediately) {
          free(is_improved_by_AMS);
          free(generational_improvement);
          return;
        }
      }
      c = 0;
      for (i = 0; i < population_sizes[population_index]; i++) {
        if (cluster_index_for_population[population_index][i] != k)
          continue;
        if (is_improved_by_AMS[i])
          updateElitistArchive(populations[population_index][i]);
        c++;
        if (c >= 0.5 * tau * num_individuals_in_cluster[population_index][k])
          break;
      }
      free(is_improved_by_AMS);
    }
  }

  for (i = 0; i < population_sizes[population_index]; i++) {
    if (generational_improvement[i])
      populations[population_index][i]->NIS = 0;
    else
      populations[population_index][i]->NIS++;
  }

  // Forced Improvements
  if (use_forced_improvement) {
    for (i = 0; i < population_sizes[population_index]; i++) {
      if (populations[population_index][i]->NIS > maximum_no_improvement_stretch) {
        applyForcedImprovements(population_index, i, &(generational_improvement[i]));

        if (global_terminate_immediately) {
          free(generational_improvement);
          return;
        }
      }
    }
  }

  cluster_failure = 1;
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
      if (distribution_multipliers[population_index][i][j] > 1.0) {
        cluster_failure = 0;
        break;
      }

  if (cluster_failure)
    no_improvement_stretch[population_index]++;

  any_improvement = 0;
  for (i = 0; i < population_sizes[population_index]; i++) {
    if (generational_improvement[i]) {
      any_improvement = 1;
      break;
    }
  }

  if (any_improvement)
    no_improvement_stretch[population_index] = 0;
  else {
    all_multipliers_leq_one = 1;
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      for (m = 0; m < linkage_model[population_index][i]->length; m++)
        if (distribution_multipliers[population_index][i][m] > 1.0) {
          all_multipliers_leq_one = 0;
          break;
        }

    if (all_multipliers_leq_one)
      no_improvement_stretch[population_index]++;
  }

  free(generational_improvement);
}

short applyAMS(int population_index, int individual_index, int cluster_index) {
  short out_of_range, improvement;
  double shrink_factor, delta_AMS, *solution_backup;
  int m;

  individual* ind_backup;
  ind_backup = initializeIndividual();

  delta_AMS = 2.0;
  out_of_range = 1;
  shrink_factor = 2;
  improvement = 0;
  solution_backup = (double*)Malloc(number_of_parameters * sizeof(double));

  copyIndividual(populations[population_index][individual_index], ind_backup);
  while ((out_of_range == 1) && (shrink_factor > 1e-10)) {
    shrink_factor *= 0.5;
    out_of_range = 0;
    for (m = 0; m < number_of_parameters; m++) {
      populations[population_index][individual_index]->parameters[m] +=
          shrink_factor * delta_AMS *
          (mean_vectors[population_index][cluster_index][m] -
           mean_vectors_previous[population_index][cluster_index][m]);
      // CHEAT
      if (use_boundary_repair) {
        if (populations[population_index][individual_index]->parameters[m] < lower_range_bounds[m])
          populations[population_index][individual_index]->parameters[m] = lower_range_bounds[m];
        else if (populations[population_index][individual_index]->parameters[m] > upper_range_bounds[m])
          populations[population_index][individual_index]->parameters[m] = upper_range_bounds[m];
      }
      // END-CHEAT
      if (!isParameterInRangeBounds(populations[population_index][individual_index]->parameters[m], m)) {
        out_of_range = 1;
        break;
      }
    }
  }
  if (!out_of_range) {
    installedProblemEvaluation(problem_index, populations[population_index][individual_index], number_of_parameters,
                               NULL, NULL, 0, 0);
    if (solutionWasImprovedByFOSElement(population_index, cluster_index, -1, individual_index) ||
        constraintParetoDominates(populations[population_index][individual_index]->objective_values,
                                  populations[population_index][individual_index]->constraint_value,
                                  ind_backup->objective_values, ind_backup->constraint_value))
      improvement = 1;
  }
  if (out_of_range || !improvement) {
    copyIndividual(ind_backup, populations[population_index][individual_index]);
  }
  free(solution_backup);
  ezilaitiniIndividual(ind_backup);
  return (improvement);
}

void applyForcedImprovements(int population_index, int individual_index, short* improved) {
  int i, j, k, m, cluster_index, donor_index, objective_index, *order, num_indices, *indices;
  double distance, distance_smallest, *objective_values_scaled, alpha, *FI_backup;
  individual* ind_backup;

  i = individual_index;
  populations[population_index][i]->NIS = 0;
  cluster_index = cluster_index_for_population[population_index][i];
  donor_index = 0;
  ind_backup = initializeIndividual();

  objective_values_scaled = (double*)Malloc(number_of_objectives * sizeof(double));
  for (j = 0; j < number_of_objectives; j++)
    objective_values_scaled[j] =
        populations[population_index][i]->objective_values[j] / objective_ranges[population_index][j];
  distance_smallest = 1e308;
  for (j = 0; j < elitist_archive_size; j++) {
    if (elitist_archive_indices_inactive[j])
      continue;
    for (k = 0; k < number_of_objectives; k++)
      objective_values_scaled[k] = elitist_archive[j]->objective_values[k] / objective_ranges[population_index][k];
    distance = distanceEuclidean(objective_values_scaled, objective_means_scaled[population_index][cluster_index],
                                 number_of_objectives);
    if (distance < distance_smallest) {
      donor_index = j;
      distance_smallest = distance;
    }
  }

  alpha = 0.5;
  while (alpha >= 0.05) {
    order = randomPermutation(linkage_model[population_index][cluster_index]->length);
    for (m = 0; m < linkage_model[population_index][cluster_index]->length; m++) {
      num_indices = linkage_model[population_index][cluster_index]->set_length[order[m]];
      indices = linkage_model[population_index][cluster_index]->sets[order[m]];

      FI_backup = (double*)Malloc(num_indices * sizeof(double));

      copyIndividualWithoutParameters(populations[population_index][i], ind_backup);

      for (j = 0; j < num_indices; j++) {
        FI_backup[j] = populations[population_index][i]->parameters[indices[j]];
        populations[population_index][i]->parameters[indices[j]] =
            alpha * populations[population_index][i]->parameters[indices[j]] +
            (1.0 - alpha) * elitist_archive[donor_index]->parameters[indices[j]];
      }
      installedProblemEvaluation(problem_index, populations[population_index][i], num_indices, indices, FI_backup,
                                 populations[population_index][i]->objective_values,
                                 populations[population_index][i]->constraint_value);
      if (global_terminate_immediately) {
        free(FI_backup);
        return;
      }

      if (single_objective_clusters[population_index][cluster_index] != -1) {
        objective_index = single_objective_clusters[population_index][cluster_index];
        if (populations[population_index][i]->objective_values[objective_index] <
            ind_backup->objective_values[objective_index])
          *improved = 1;
      } else if (constraintParetoDominates(populations[population_index][i]->objective_values,
                                           populations[population_index][i]->constraint_value,
                                           ind_backup->objective_values, ind_backup->constraint_value))
        *improved = 1;

      if (!(*improved)) {
        for (j = 0; j < num_indices; j++)
          populations[population_index][i]->parameters[indices[j]] = FI_backup[j];
        copyIndividualWithoutParameters(ind_backup, populations[population_index][i]);
        free(FI_backup);
      } else {
        free(FI_backup);
        break;
      }
    }
    alpha *= 0.5;

    free(order);
    if (*improved)
      break;
  }
  if (!(*improved)) {
    copyIndividual(elitist_archive[donor_index], populations[population_index][i]);
  }
  updateElitistArchive(populations[population_index][i]);
  ezilaitiniIndividual(ind_backup);

  free(objective_values_scaled);
}

/**
 * Computes the Cholesky-factor matrices required for sampling
 * the multivariate normal distributions in the mixture distribution.
 */
void computeParametersForSampling(int population_index, int cluster_index) {
  int i;

  if (!use_univariate_FOS) {
    decomposed_cholesky_factors_lower_triangle[population_index][cluster_index] =
        (double***)Malloc(linkage_model[population_index][cluster_index]->length * sizeof(double**));
    for (i = 0; i < linkage_model[population_index][cluster_index]->length; i++)
      decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][i] =
          choleskyDecomposition(decomposed_covariance_matrices[population_index][cluster_index][i],
                                linkage_model[population_index][cluster_index]->set_length[i]);
  }
}

/**
 * Generates and returns a single new solution by drawing
 * a sample for the variables in the selected FOS elementmax_clus
 * and inserting this into the population.
 */
double* generateNewPartialSolutionFromFOSElement(int population_index, int cluster_index, int FOS_index) {
  short ready;
  int i, times_not_in_bounds, num_indices, *indices;
  double *result, *z;

  num_indices = linkage_model[population_index][cluster_index]->set_length[FOS_index];
  indices = linkage_model[population_index][cluster_index]->sets[FOS_index];
  times_not_in_bounds = -1;
  out_of_bounds_draws[population_index][cluster_index][FOS_index]--;

  ready = 0;
  do {
    times_not_in_bounds++;
    samples_drawn_from_normal[population_index][cluster_index][FOS_index]++;
    out_of_bounds_draws[population_index][cluster_index][FOS_index]++;
    if (times_not_in_bounds >= 100) {
      result = (double*)Malloc(num_indices * sizeof(double));
      for (i = 0; i < num_indices; i++)
        result[i] = lower_init_ranges[indices[i]] +
                    (upper_init_ranges[indices[i]] - lower_init_ranges[indices[i]]) * randomRealUniform01();
    } else {
      z = (double*)Malloc(num_indices * sizeof(double));

      for (i = 0; i < num_indices; i++)
        z[i] = random1DNormalUnit();

      if (use_univariate_FOS) {
        result = (double*)Malloc(1 * sizeof(double));
        result[0] = z[0] * sqrt(decomposed_covariance_matrices[population_index][cluster_index][FOS_index][0][0]) +
                    mean_vectors[population_index][cluster_index][indices[0]];
      } else {
        result = matrixVectorMultiplication(
            decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][FOS_index], z, num_indices,
            num_indices);

        for (i = 0; i < num_indices; i++)
          result[i] += mean_vectors[population_index][cluster_index][indices[i]];
      }

      free(z);
    }

    ready = 1;
    for (i = 0; i < num_indices; i++) {
      // CHEAT
      if (use_boundary_repair) {
        if (result[i] < lower_range_bounds[indices[i]])
          result[i] = lower_range_bounds[indices[i]];
        else if (result[i] > upper_range_bounds[indices[i]])
          result[i] = upper_range_bounds[indices[i]];
      }
      // END-CHEAT

      if (!isParameterInRangeBounds(result[i], indices[i])) {
        ready = 0;
        break;
      }
    }
    if (!ready)
      free(result);
  } while (!ready);

  return (result);
}

/**
 * Generates and returns a single new solution by drawing
 * a single sample from a specified model.
 */
short generateNewSolutionFromFOSElement(int population_index, int cluster_index, int FOS_index, int individual_index) {
  int j, m, *indices, num_indices, *touched_indices, num_touched_indices, out_of_range;
  double *result, *solution_AMS, *individual_backup, shrink_factor;
  short improvement;
  individual* ind_backup;
  ind_backup = initializeIndividual();

  num_indices = linkage_model[population_index][cluster_index]->set_length[FOS_index];
  indices = linkage_model[population_index][cluster_index]->sets[FOS_index];
  num_touched_indices = num_indices;
  touched_indices = indices;
  improvement = 0;

  solution_AMS = (double*)Malloc(num_indices * sizeof(double));
  individual_backup = (double*)Malloc(num_touched_indices * sizeof(double));

  result = generateNewPartialSolutionFromFOSElement(population_index, cluster_index, FOS_index);

  for (j = 0; j < num_touched_indices; j++)
    individual_backup[j] = populations[population_index][individual_index]->parameters[touched_indices[j]];
  for (j = 0; j < num_indices; j++)
    populations[population_index][individual_index]->parameters[indices[j]] = result[j];

  copyIndividualWithoutParameters(populations[population_index][individual_index], ind_backup);

  if ((number_of_generations[population_index] > 0) &&
      (samples_current_cluster <= 0.5 * tau * num_individuals_in_cluster[population_index][cluster_index])) {
    out_of_range = 1;
    shrink_factor = 2;
    while ((out_of_range == 1) && (shrink_factor > 1e-10)) {
      shrink_factor *= 0.5;
      out_of_range = 0;
      for (m = 0; m < num_indices; m++) {
        j = indices[m];
        solution_AMS[m] = result[m] + shrink_factor * delta_AMS *
                                          distribution_multipliers[population_index][cluster_index][FOS_index] *
                                          (mean_vectors[population_index][cluster_index][j] -
                                           mean_vectors_previous[population_index][cluster_index][j]);
        // CHEAT
        if (use_boundary_repair) {
          if (solution_AMS[m] < lower_range_bounds[indices[m]])
            solution_AMS[m] = lower_range_bounds[indices[m]];
          else if (solution_AMS[m] > upper_range_bounds[indices[m]])
            solution_AMS[m] = upper_range_bounds[indices[m]];
        }
        // END-CHEAT
        if (!isParameterInRangeBounds(solution_AMS[m], j)) {
          out_of_range = 1;
          break;
        }
      }
    }
    if (!out_of_range) {
      for (j = 0; j < num_indices; j++)
        populations[population_index][individual_index]->parameters[indices[j]] = solution_AMS[j];
    }
  }
  installedProblemEvaluation(problem_index, populations[population_index][individual_index], num_touched_indices,
                             touched_indices, individual_backup, ind_backup->objective_values,
                             ind_backup->constraint_value);

  if (solutionWasImprovedByFOSElement(population_index, cluster_index, FOS_index, individual_index) ||
      constraintParetoDominates(populations[population_index][individual_index]->objective_values,
                                populations[population_index][individual_index]->constraint_value,
                                ind_backup->objective_values, ind_backup->constraint_value)) {
    improvement = 1;
  }

  if (!improvement) {
    for (j = 0; j < num_touched_indices; j++)
      populations[population_index][individual_index]->parameters[touched_indices[j]] = individual_backup[j];
    copyIndividualWithoutParameters(ind_backup, populations[population_index][individual_index]);
  }

  free(solution_AMS);
  free(individual_backup);
  free(result);

  ezilaitiniIndividual(ind_backup);
  return (improvement);
}

/**
 * Adapts the distribution multipliers according to
 * the SDR-AVS mechanism.
 */
void adaptDistributionMultipliers(int population_index, int cluster_index, int FOS_index) {
  short improvementForFOSElement;
  double st_dev_ratio;

  if ((((double)out_of_bounds_draws[population_index][cluster_index][FOS_index]) /
       ((double)samples_drawn_from_normal[population_index][cluster_index][FOS_index])) > 0.9) {
    distribution_multipliers[population_index][cluster_index][FOS_index] *= 0.5;
  }

  improvementForFOSElement =
      generationalImprovementForOneClusterForFOSElement(population_index, cluster_index, FOS_index, &st_dev_ratio);

  if (improvementForFOSElement) {
    no_improvement_stretch[population_index] = 0;

    if (distribution_multipliers[population_index][cluster_index][FOS_index] < 1.0)
      distribution_multipliers[population_index][cluster_index][FOS_index] = 1.0;

    if (st_dev_ratio > st_dev_ratio_threshold)
      distribution_multipliers[population_index][cluster_index][FOS_index] *= distribution_multiplier_increase;
  } else {
    if ((distribution_multipliers[population_index][cluster_index][FOS_index] > 1.0) ||
        (no_improvement_stretch[population_index] >= maximum_no_improvement_stretch))
      distribution_multipliers[population_index][cluster_index][FOS_index] *= distribution_multiplier_decrease;

    if (no_improvement_stretch[population_index] < maximum_no_improvement_stretch) {
      if (distribution_multipliers[population_index][cluster_index][FOS_index] < 1.0)
        distribution_multipliers[population_index][cluster_index][FOS_index] = 1.0;
    }
  }
}

/**
 * Determines whether an improvement is found for a specified
 * population. Returns 1 in case of an improvement, 0 otherwise.
 * The standard-deviation ratio required by the SDR-AVS
 * mechanism is computed and returned in the pointer variable.
 */
short generationalImprovementForOneClusterForFOSElement(int population_index,
                                                        int cluster_index,
                                                        int FOS_index,
                                                        double* st_dev_ratio) {
  int i, number_of_improvements;

  number_of_improvements = 0;

  /* Determine st.dev. ratio */
  *st_dev_ratio = 0.0;
  for (i = 0; i < population_sizes[population_index]; i++) {
    if (cluster_index_for_population[population_index][i] == cluster_index) {
      if (solutionWasImprovedByFOSElement(population_index, cluster_index, FOS_index, i)) {
        number_of_improvements++;
        (*st_dev_ratio) += getStDevRatioForOneClusterForFOSElement(population_index, cluster_index, FOS_index,
                                                                   populations[population_index][i]->parameters);
      }
    }
  }

  if (number_of_improvements > 0)
    (*st_dev_ratio) = (*st_dev_ratio) / number_of_improvements;

  if (number_of_improvements > 0)
    return (1);

  return (0);
}

/**
 * Computes and returns the standard-deviation-ratio
 * of a given point for a given model.
 */
double getStDevRatioForOneClusterForFOSElement(int population_index,
                                               int cluster_index,
                                               int FOS_index,
                                               double* parameters) {
  int i, *indices, num_indices;
  double **inverse, result, *x_min_mu, *z;

  result = 0.0;
  indices = linkage_model[population_index][cluster_index]->sets[FOS_index];
  num_indices = linkage_model[population_index][cluster_index]->set_length[FOS_index];

  x_min_mu = (double*)Malloc(num_indices * sizeof(double));

  for (i = 0; i < num_indices; i++)
    x_min_mu[i] = parameters[indices[i]] - mean_vectors[population_index][cluster_index][indices[i]];

  if (use_univariate_FOS) {
    result = fabs(x_min_mu[0] / sqrt(decomposed_covariance_matrices[population_index][cluster_index][FOS_index][0][0]));
  } else {
    inverse = matrixLowerTriangularInverse(
        decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][FOS_index], num_indices);
    z = matrixVectorMultiplication(inverse, x_min_mu, num_indices, num_indices);

    for (i = 0; i < num_indices; i++) {
      if (fabs(z[i]) > result)
        result = fabs(z[i]);
    }

    free(z);
    for (i = 0; i < num_indices; i++)
      free(inverse[i]);
    free(inverse);
  }

  free(x_min_mu);

  return (result);
}

/**
 * Returns whether a solution has the
 * hallmark of an improvement (1 for yes, 0 for no).
 */
short solutionWasImprovedByFOSElement(int population_index, int cluster_index, int FOS_index, int individual_index) {
  short result, in_range;
  int i, j;

  result = 0;
  in_range = 1;
  if (FOS_index == -1) {
    for (i = 0; i < number_of_parameters; i++)
      if (!isParameterInRangeBounds(populations[population_index][individual_index]->parameters[i], i))
        return (result);
  } else
    in_range = isSolutionInRangeBoundsForFOSElement(populations[population_index][individual_index]->parameters,
                                                    population_index, cluster_index, FOS_index);
  if (in_range) {
    if (populations[population_index][individual_index]->constraint_value == 0) {
      for (j = 0; j < number_of_objectives; j++) {
        if (populations[population_index][individual_index]->objective_values[j] <
            best_objective_values_in_elitist_archive[j]) {
          result = 1;
          break;
        }
      }
    }

    if (single_objective_clusters[population_index][cluster_index] != -1) {
      return (result);
    }

    if (result != 1) {
      result = 1;
      for (i = 0; i < elitist_archive_size; i++) {
        if (elitist_archive_indices_inactive[i])
          continue;
        if (constraintParetoDominates(elitist_archive[i]->objective_values, elitist_archive[i]->constraint_value,
                                      populations[population_index][individual_index]->objective_values,
                                      populations[population_index][individual_index]->constraint_value)) {
          result = 0;
          break;
        } else if (!constraintParetoDominates(populations[population_index][individual_index]->objective_values,
                                              populations[population_index][individual_index]->constraint_value,
                                              elitist_archive[i]->objective_values,
                                              elitist_archive[i]->constraint_value)) {
          if (sameObjectiveBox(elitist_archive[i]->objective_values,
                               populations[population_index][individual_index]->objective_values)) {
            result = 0;
            break;
          }
        }
      }
    }
  }

  return (result);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Ezilaitini -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Undoes initialization procedure by freeing up memory.
 */
void ezilaitini(void) {
  int i;

  ezilaitiniObjectiveRotationMatrix();

  for (i = 0; i < number_of_populations; i++) {
    ezilaitiniDistributionMultipliers(i);

    ezilaitiniMemoryOnePopulation(i);
  }

  ezilaitiniMemory();

  ezilaitiniProblem();
}

void ezilaitiniMemory(void) {
  int i, default_front_size;
  double** default_front;

  for (i = 0; i < elitist_archive_capacity; i++)
    ezilaitiniIndividual(elitist_archive[i]);
  free(elitist_archive);

  if (use_vtr) {
    default_front = getDefaultFront(&default_front_size);
    if (default_front) {
      for (i = 0; i < default_front_size; i++)
        free(default_front[i]);
      free(default_front);
    }
  }

  free(full_covariance_matrix);
  free(population_sizes);
  free(selection_sizes);
  free(cluster_sizes);
  free(populations);
  free(ranks);
  free(sorted_ranks);
  free(objective_ranges);
  free(selection);
  free(objective_values_selection_previous);
  free(ranks_selection);
  free(number_of_mixing_components);
  free(decomposed_covariance_matrices);
  free(distribution_multipliers);
  free(decomposed_cholesky_factors_lower_triangle);
  free(mean_vectors);
  free(mean_vectors_previous);
  free(objective_means_scaled);
  free(selection_indices);
  free(selection_indices_of_cluster_members);
  free(selection_indices_of_cluster_members_previous);
  free(pop_indices_selected);
  free(samples_drawn_from_normal);
  free(out_of_bounds_draws);
  free(single_objective_clusters);
  free(cluster_index_for_population);
  free(num_individuals_in_cluster);
  free(number_of_generations);
  free(populations_terminated);
  free(no_improvement_stretch);

  free(lower_range_bounds);
  free(upper_range_bounds);
  free(lower_init_ranges);
  free(upper_init_ranges);

  free(number_of_elitist_solutions_copied);
  free(best_objective_values_in_elitist_archive);
  free(elitist_archive_indices_inactive);
  free(objective_discretization);

  free(linkage_model);
}

/**
 * Undoes initialization procedure by freeing up memory.
 */
void ezilaitiniMemoryOnePopulation(int population_index) {
  int i, j;

  for (i = 0; i < population_sizes[population_index]; i++) {
    ezilaitiniIndividual(populations[population_index][i]);
  }
  free(populations[population_index]);

  for (i = 0; i < selection_sizes[population_index]; i++) {
    ezilaitiniIndividual(selection[population_index][i]);
    free(objective_values_selection_previous[population_index][i]);
  }
  free(selection[population_index]);
  free(objective_values_selection_previous[population_index]);

  if (!learn_linkage_tree) {
    ezilaitiniCovarianceMatrices(population_index);
  }

  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    free(mean_vectors[population_index][i]);
    free(mean_vectors_previous[population_index][i]);
    free(objective_means_scaled[population_index][i]);

    if (selection_indices_of_cluster_members[population_index][i] != NULL)
      free(selection_indices_of_cluster_members[population_index][i]);
    if (selection_indices_of_cluster_members_previous[population_index][i] != NULL)
      free(selection_indices_of_cluster_members_previous[population_index][i]);

    if (samples_drawn_from_normal[population_index] != NULL) {
      free(samples_drawn_from_normal[population_index][i]);
      free(out_of_bounds_draws[population_index][i]);
    }

    if (linkage_model[population_index][i] != NULL) {
      for (j = 0; j < linkage_model[population_index][i]->length; j++)
        free(linkage_model[population_index][i]->sets[j]);
      free(linkage_model[population_index][i]->sets);
      free(linkage_model[population_index][i]->set_length);
      free(linkage_model[population_index][i]);
    }
  }

  if (learn_linkage_tree) {
    free(full_covariance_matrix[population_index]);
  }

  free(linkage_model[population_index]);
  free(ranks[population_index]);
  free(sorted_ranks[population_index]);
  free(objective_ranges[population_index]);
  free(ranks_selection[population_index]);
  free(mean_vectors[population_index]);
  free(mean_vectors_previous[population_index]);
  free(objective_means_scaled[population_index]);
  free(selection_indices[population_index]);
  free(selection_indices_of_cluster_members[population_index]);
  free(selection_indices_of_cluster_members_previous[population_index]);
  free(pop_indices_selected[population_index]);
  free(decomposed_cholesky_factors_lower_triangle[population_index]);
  free(samples_drawn_from_normal[population_index]);
  free(out_of_bounds_draws[population_index]);
  free(single_objective_clusters[population_index]);
  free(cluster_index_for_population[population_index]);
  free(num_individuals_in_cluster[population_index]);
  free(number_of_elitist_solutions_copied[population_index]);
}

/**
 * Undoes initialization procedure by freeing up memory.
 */
void ezilaitiniDistributionMultipliers(int population_index) {
  int i;
  if (distribution_multipliers[population_index] == NULL)
    return;

  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    free(distribution_multipliers[population_index][i]);
  }
  free(distribution_multipliers[population_index]);
}

void ezilaitiniCovarianceMatrices(int population_index) {
  int i, j, k;

  for (i = 0; i < number_of_mixing_components[population_index]; i++) {
    for (j = 0; j < linkage_model[population_index][i]->length; j++) {
      for (k = 0; k < linkage_model[population_index][i]->set_length[j]; k++)
        free(decomposed_covariance_matrices[population_index][i][j][k]);
      free(decomposed_covariance_matrices[population_index][i][j]);
    }
    free(decomposed_covariance_matrices[population_index][i]);
  }
  free(decomposed_covariance_matrices[population_index]);
}

/**
 * Frees memory of the Cholesky decompositions required for sampling.
 */
void ezilaitiniParametersForSampling(int population_index) {
  int i, j, k;

  if (!use_univariate_FOS) {
    for (k = 0; k < number_of_mixing_components[population_index]; k++) {
      for (i = 0; i < linkage_model[population_index][k]->length; i++) {
        for (j = 0; j < linkage_model[population_index][k]->set_length[i]; j++)
          free(decomposed_cholesky_factors_lower_triangle[population_index][k][i][j]);
        free(decomposed_cholesky_factors_lower_triangle[population_index][k][i]);
      }
      free(decomposed_cholesky_factors_lower_triangle[population_index][k]);
    }
  }
  if (learn_linkage_tree) {
    ezilaitiniCovarianceMatrices(population_index);
  }
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Run -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

void generationalStepAllPopulationsRecursiveFold(int population_index_smallest, int population_index_biggest);
void generationalStepAllPopulations() {
  int population_index_smallest, population_index_biggest;

  population_index_biggest = number_of_populations - 1;
  population_index_smallest = 0;
  while (population_index_smallest <= population_index_biggest) {
    if (!populations_terminated[population_index_smallest])
      break;

    population_index_smallest++;
  }

  generationalStepAllPopulationsRecursiveFold(population_index_smallest, population_index_biggest);
}

void generationalStepAllPopulationsRecursiveFold(int population_index_smallest, int population_index_biggest) {
  int i, j, population_index;

  for (i = 0; i < number_of_subgenerations_per_population_factor - 1; i++) {
    for (population_index = population_index_smallest; population_index <= population_index_biggest;
         population_index++) {
      if (!populations_terminated[population_index]) {
        makeSelection(population_index);

        makePopulation(population_index);

        (number_of_generations[population_index])++;

        if (checkTerminationConditionOnePopulation(population_index)) {
          for (j = 0; j < number_of_populations; j++)
            populations_terminated[j] = 1;
          return;
        }
      }
    }

    for (population_index = population_index_smallest; population_index < population_index_biggest; population_index++)
      generationalStepAllPopulationsRecursiveFold(population_index_smallest, population_index);
  }
}

void runAllPopulations(void) {
  while (!checkTerminationConditionAllPopulations()) {
    if (number_of_populations < maximum_number_of_populations) {
      initializeNewPopulation();
    }

    if (global_terminate_immediately) {
      return;
    }

    /*
    computeApproximationSet();

    if (write_generational_statistics)
      writeGenerationalStatisticsForOnePopulation(number_of_populations - 1);

    if (write_generational_solutions)
      writeGenerationalSolutions(0);

    freeApproximationSet();
     */

    generationalStepAllPopulations();

    total_number_of_generations++;
  }
}

/*
void run(void) {
  initialize();

  if (print_verbose_overview)
    printVerboseOverview();

  runAllPopulations();

  computeApproximationSet();

  writeGenerationalStatisticsForOnePopulation(number_of_populations - 1);

  writeGenerationalSolutions(1);
  freeApproximationSet();

  ezilaitini();
}


int main(int argc, char** argv) {
  initializeRandomNumberGenerator();

  interpretCommandLine(argc, argv);

  run();

  return (0);
}

*/

};  // namespace mo_rv_gomea_impl

namespace goblin {

std::optional<u64> MoRvGOMEA::current_generation() const {
  // DEADLOCK -> reads are fine I guess...
  // const std::lock_guard<std::mutex> lock(mo_rv_gomea_impl::global_instance_mutex);
  u64 g = 0;
  for (usize i = 0; i < mo_rv_gomea_impl::number_of_populations; i++) {
    g += static_cast<u64>(mo_rv_gomea_impl::number_of_generations[i]);
  }
  return g;
};

std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> MoRvGOMEA::run(InstanceBase& problem,
                                                                           const Budget& budget,
                                                                           std::optional<u64> seed,
                                                                           std::optional<usize> population_size) {
  if (problem.num_discrete() > 0 || problem.num_continuous() < 1) {
    throw std::runtime_error("Discrete/Mixed problem types are not supported by MO-RV-GOMEA");
  }

  const std::lock_guard<std::mutex> lock(mo_rv_gomea_impl::global_instance_mutex);

  // set algorithm parameters
  mo_rv_gomea_impl::maximum_number_of_evaluations = budget.max_evaluations.value_or(std::numeric_limits<int>::max());
  mo_rv_gomea_impl::maximum_number_of_seconds = std::numeric_limits<double>::max();
  if (budget.max_time.has_value()) {
    std::chrono::duration<double> max_seconds = budget.max_time.value();
    mo_rv_gomea_impl::maximum_number_of_seconds = max_seconds.count();
  }

  mo_rv_gomea_impl::write_generational_statistics = 0;
  mo_rv_gomea_impl::write_generational_solutions = 0;
  mo_rv_gomea_impl::print_verbose_overview = 0;
  mo_rv_gomea_impl::use_vtr = 0;
  mo_rv_gomea_impl::use_guidelines = 0;
  mo_rv_gomea_impl::static_linkage_tree = 0;
  mo_rv_gomea_impl::learn_linkage_tree = 0;
  mo_rv_gomea_impl::rotation_angle = 0.0;

  mo_rv_gomea_impl::tau = selection_percentile;
  mo_rv_gomea_impl::st_dev_ratio_threshold = std_deviation_ratio_threshold;
  mo_rv_gomea_impl::elitist_archive_size_target = target_archive_size;

  mo_rv_gomea_impl::black_box_evaluations = partial_evaluations ? 0 : 1;
  mo_rv_gomea_impl::use_boundary_repair = boundary_repair ? 1 : 0;
  mo_rv_gomea_impl::use_forced_improvement = forced_improvements ? 1 : 0;

  mo_rv_gomea_impl::problem_index = 10;
  mo_rv_gomea_impl::number_of_objectives = problem.num_objectives();
  mo_rv_gomea_impl::number_of_parameters = problem.num_continuous();
  mo_rv_gomea_impl::lower_user_range = problem.continuous_init_lower_bounds().minCoeff();
  mo_rv_gomea_impl::upper_user_range = problem.continuous_init_upper_bounds().maxCoeff();

  if (population_size.has_value()) {
    mo_rv_gomea_impl::base_population_size = population_size.value();
    mo_rv_gomea_impl::maximum_number_of_populations = 1;

    mo_rv_gomea_impl::base_number_of_mixing_components =
        initial_num_clusters.value_or(1 + mo_rv_gomea_impl::number_of_objectives);
  } else {
    mo_rv_gomea_impl::maximum_number_of_populations = max_num_populations;

    if (mo_rv_gomea_impl::maximum_number_of_populations > 1) {
      mo_rv_gomea_impl::base_number_of_mixing_components =
          initial_num_clusters.value_or(1 + mo_rv_gomea_impl::number_of_objectives);
      mo_rv_gomea_impl::base_population_size =
          initial_population_size.value_or(10 * mo_rv_gomea_impl::base_number_of_mixing_components);
    } else {
      mo_rv_gomea_impl::base_number_of_mixing_components = initial_num_clusters.value_or(20);

      mo_rv_gomea_impl::base_population_size =
          initial_population_size.value_or((int)((0.5 * mo_rv_gomea_impl::base_number_of_mixing_components) *
                                                 (36.1 + 7.58 * log2((double)mo_rv_gomea_impl::number_of_parameters))));
    }
  }

  if (static_cast<int>(mo_rv_gomea_impl::tau * static_cast<double>(mo_rv_gomea_impl::base_population_size)) <= 0) {
    throw std::runtime_error("Selection percentile must be in [1/initial_population_size, 1].");
  }

  mo_rv_gomea_impl::number_of_subgenerations_per_population_factor = subgeneration_factor;
  mo_rv_gomea_impl::maximum_no_improvement_stretch = max_no_improvement_stretch.value_or(
      (int)(2.0 + ((double)(25 + mo_rv_gomea_impl::number_of_parameters)) /
                      ((double)mo_rv_gomea_impl::base_number_of_mixing_components)));

  mo_rv_gomea_impl::statistics_file_existed = 0;
  mo_rv_gomea_impl::objective_discretization_in_effect = 0;
  mo_rv_gomea_impl::block_size = mo_rv_gomea_impl::number_of_parameters;
  mo_rv_gomea_impl::block_start = 0;
  // if (mo_rv_gomea_impl::problem_index == 9) {
  //   mo_rv_gomea_impl::block_size = 5;
  //   mo_rv_gomea_impl::block_start = 1;
  // }
  mo_rv_gomea_impl::number_of_blocks =
      (mo_rv_gomea_impl::number_of_parameters - 1 + mo_rv_gomea_impl::block_size - 1) / mo_rv_gomea_impl::block_size;

  // init FOS
  mo_rv_gomea_impl::FOS_element_ub = max_subset_size.value_or(mo_rv_gomea_impl::number_of_parameters);
  mo_rv_gomea_impl::global_static_fos = NULL;
  if (std::holds_alternative<StaticFOS>(linkage_model)) {
    mo_rv_gomea_impl::global_static_fos = &std::get<StaticFOS>(linkage_model);
  } else {
    std::string lm = std::get<std::string>(linkage_model);

    if (lm == "Full") {
      mo_rv_gomea_impl::FOS_element_size = -1;
      mo_rv_gomea_impl::FOS_element_size = mo_rv_gomea_impl::number_of_parameters;
    } else if (lm == "LinkageTree") {
      mo_rv_gomea_impl::FOS_element_size = -2;
      mo_rv_gomea_impl::learn_linkage_tree = 1;
    } else if (lm == "StaticLinkageTree") {
      mo_rv_gomea_impl::FOS_element_size = -3;
      mo_rv_gomea_impl::static_linkage_tree = 1;
    } else if (lm == "BFLT") {
      mo_rv_gomea_impl::FOS_element_size = -4;
      mo_rv_gomea_impl::static_linkage_tree = 1;
      mo_rv_gomea_impl::FOS_element_ub = 100;
    } else if (lm == "RandomBFLT") {
      mo_rv_gomea_impl::FOS_element_size = -5;
      mo_rv_gomea_impl::random_linkage_tree = 1;
      mo_rv_gomea_impl::static_linkage_tree = 1;
      mo_rv_gomea_impl::FOS_element_ub = 100;
    } else if (lm == "RandomLinkageTree") {
      mo_rv_gomea_impl::FOS_element_size = -6;
      mo_rv_gomea_impl::random_linkage_tree = 1;
    } else if (lm == "StaticRandomLinkageTree") {
      mo_rv_gomea_impl::FOS_element_size = -7;
      mo_rv_gomea_impl::random_linkage_tree = 1;
      mo_rv_gomea_impl::static_linkage_tree = 1;
    } else if (lm == "Univariate") {
      mo_rv_gomea_impl::FOS_element_size = 1;
      mo_rv_gomea_impl::use_univariate_FOS = 1;
    } else {
      throw std::runtime_error("Unknown or unsupported linkage model: '" + lm + "'");
    }
  }

  // initialize global library resources
  auto archive = std::make_shared<AdaptiveGridArchive>(problem.archive_fitness(), target_archive_size);
  mo_rv_gomea_impl::global_status = TerminationStatus::Running;
  mo_rv_gomea_impl::global_problem_ptr = &problem;
  mo_rv_gomea_impl::global_archive_ptr = archive.get();
  mo_rv_gomea_impl::global_solution_set.clear();
  mo_rv_gomea_impl::global_solution_set.add(
      Solution(problem.archive_fitness().worst(), std::nullopt, Vec<CType>::Zero(problem.num_continuous())));
  mo_rv_gomea_impl::global_parent_set.clear();
  mo_rv_gomea_impl::global_parent_set.add(mo_rv_gomea_impl::global_solution_set[0]);

  // init MO-RV-GOMEA and library rngs
  mo_rv_gomea_impl::random_seed_changing = seed.value_or(0);
  mo_rv_gomea_impl::initializeRandomNumberGenerator();
  Rng rng = seeded_rng(seed);
  mo_rv_gomea_impl::global_rng_ptr = &rng;

  // actually run MO-RV-GOMEA
  mo_rv_gomea_impl::global_terminate_immediately = false;
  mo_rv_gomea_impl::startTimer();

  // mo_rv_gomea_impl::checkOptions();
  mo_rv_gomea_impl::initialize();
  mo_rv_gomea_impl::runAllPopulations();
  // mo_rv_gomea_impl::computeApproximationSet();

  // free memory
  // mo_rv_gomea_impl::freeApproximationSet(); // only needed for MO-RV-GOMEA internal logging which is disabled
  mo_rv_gomea_impl::ezilaitini();

  // TODO this does not work as intended and also show up for e.g. budget exhaustion (if the algorithm was not stopped
  // due to success/budget exhaustion, some internal convergence check probably triggered)
  if (mo_rv_gomea_impl::global_status == TerminationStatus::Running) {
    mo_rv_gomea_impl::global_status = TerminationStatus::Converged;
  }

  // reset non-owning resource pointers
  mo_rv_gomea_impl::global_rng_ptr = NULL;
  mo_rv_gomea_impl::global_problem_ptr = NULL;
  mo_rv_gomea_impl::global_archive_ptr = NULL;
  mo_rv_gomea_impl::global_static_fos = NULL;

  return std::make_tuple(archive, mo_rv_gomea_impl::global_status);
}

};  // namespace goblin
