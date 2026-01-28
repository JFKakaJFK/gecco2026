# Solutions

A solution at the minimum needs to store the decision variables to optimize and the quality or fitness thereof. To enable common solution types across algorithms and domains, the solution type holds all supported types of decision variables, and quality types are polymorphic (i.e. anything goes). In addition, solutions can hold additional extensions, for example to support domain specific mechanisms that are stateful or to track additional statistics.
