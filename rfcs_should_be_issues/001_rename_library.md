# GOBLIN -> GOMEA

The library name needs to be changed.
Why? Who wants to be associated with goblins?

How:
- rename pygom/pygoblin to gomea
- rename the root namespace goblin to gomea - While gomea is sort of taken by the previous library, that only affects the wrappers. There, ideally the headers/interface should not need to know about the old library and the following trick can be used to avoid namespace collisions:

```C++

namespace detail { // wrapping the include in a namespace wraps all types as well...
    #include <old_gomea_library.h>
};
    
namespace gomea {
    detail::gomea::old_lib_type;
};
```
