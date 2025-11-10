#pragma once

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "gomea/src/common/gomea_defs.hpp"
#include "gomea/src/common/solution.hpp"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace gomea{
namespace realvalued{

template<class T>
class solution_t : public gomea::solution_t<T>
{
	using gomea::solution_t<T>::solution_t;

	public:
		int NIS = 0; // no improvement stretch
};

}}
