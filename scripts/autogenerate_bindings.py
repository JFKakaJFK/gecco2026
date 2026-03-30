import os
import pathlib
import re

import litgen
from codemanip import amalgamated_header

REPO_DIR = pathlib.Path(os.path.realpath(os.path.dirname(__file__) + "/../"))
INCLUDE_DIR = REPO_DIR / "lib" / "include"
PYLIB_DIR = REPO_DIR / "pylib"

os.environ["GOBLIN_BUILD_PYLIB"] = "ON"


# from https://github.com/pthom/litgen/blob/27ee14babe490467dafbc274d654794b314ed777/src/litgen/internal/cpp_to_python.py#L659
def type_replacements() -> litgen.RegexReplacementList:
    """Replacements for C++ code when translating to python.

    Consists mostly of
    * types translations
    * NULL, nullptr, void translation
    """

    replacements_str = r"""
    \bunsigned \s*int\b -> int
    \bunsigned \s*short\b -> int
    \bunsigned \s*long long\b -> int
    \bunsigned \s*long\b -> int
    \buint8_t\b -> int
    \bint8_t\b -> int
    \buint16_t\b -> int
    \bint16_t\b -> int
    \buint32_t\b -> int
    \bint32_t\b -> int
    \buint64_t\b -> int
    \bint64_t\b -> int
    \blong\b -> int
    \bshort\b -> int
    \\blong \s*long\b -> int
    \blong \s*long\b -> int

    \blong \s*double\b -> float
    \bdouble\b -> float
    \bfloat\b -> float

    \bconst \s*char*\b -> str
    \bconst \s*char *\b -> str

    \bsize_t\b -> int
    \bstd::function<(.*)\((.*)\)> -> Callable[[\2], \1]
    \bstd::string\(\) -> ""
    \bstd::string\b -> str
    \bstd::unique_ptr<(.*?)> -> \1
    \bstd::shared_ptr<(.*?)> -> \1
    \bstd::vector\s*<\s*(.*?)\s*> -> List[\1]
    \bstd::array\s*<\s*(.*?)\s*,\s*(.*?)\s*> -> List[\1]
    \bstd::tuple<(.*?)> -> Tuple[\1]
    \bstd::pair<(.*?)> -> Tuple[\1]
    \bstd::variant<(.*?)> -> Union[\1]
    \bstd::optional<(.*?)> -> Optional[\1]
    \bstd::map<\s*(.*?)\s*,\s*(.*?)\s*> -> Dict[\1, \2]

    \bvoid\s*\* -> Any
    \bvoid\b -> None

    \bpy::array\b -> np.ndarray
    \bpy::ndarray<(.*?)> -> np.ndarray
    \bnb::array\b -> np.ndarray
    \bnb::ndarray<(.*?)> -> np.ndarray

    \bconst\b -> REMOVE
    \bmutable\b -> REMOVE
    & -> REMOVE
    \* -> REMOVE
    """

    # TODO this doesn't really do the job - it might be necessary to manually replace some std:: and custom definitions after litgen is done...
    customm_replacements_str = r"""
    \bVec<(.*)>\b -> np.ndarray
    \bArray<(.*)>\b -> np.ndarray
    \bMat<(.*)>\b -> np.ndarray
    \bArr2D<(.*)>\b -> np.ndarray
    \bC?RefS?<(.*)>\b -> np.ndarray

    \bVec\[(.*)\]\b -> np.ndarray
    \bArray\[(.*)\]\b -> np.ndarray
    \bMat\[(.*)\]\b -> np.ndarray
    \bArr2D\[(.*)\]\b -> np.ndarray
    \bC?RefS?\[(.*)\]\b -> np.ndarray

    \bu8\b -> int
    \bu16\b -> int
    \bu32\b -> int
    \bu64\b -> int
    \busize\b -> int

    \bf32\b -> float
    \bf64\b -> float

    \bBType\b -> bool
    \bDType\b -> int
    \bCType\b -> float

    \bstd::ostream\b -> io.IOBase
    \bstd::string_view\b -> str
    \bstd::reference_wrapper<(.*)>\b -> \1
    \bstd\.reference_wrapper\[(.*)\]\b -> \1
    \bstd::set<(.*)>\b -> Set[\1]
    \bstd::make_shared<(.*)>\b -> \1
    \bstd::make_unique<(.*)>\b -> \1
    \bstd::filesystem::path\b -> pathlib.Path | str
    \bstd::chrono::nanoseconds\b -> datetime.timedelta
    \bstd::span<(.*)>\b -> List[\1]
    \bstd::chrono::seconds\((\d+)\)\b -> datetime.timedelta(seconds=\1)
    \bstd::chrono::minutes\((\d+)\)\b -> datetime.timedelta(minutes=\1)
    """

    replaces = litgen.RegexReplacementList.from_string(
        replacements_str + customm_replacements_str
    )
    return replaces


def postprocess_stub(code: str):
    patterns = [
        tuple(map(lambda t: t.strip(), s.split("->")))
        for r in r"""
        Vec\[([^\]]*)\] -> np.ndarray
        # negative lookbehind to avoid looping
        (?<!npt\.ND)Array\[([^\]]*)\] -> np.ndarray
        Mat\[([^\]]*)\] -> np.ndarray
        Arr2D\[([^\]]*)\] -> np.ndarray
        C?RefS?\[([^\]]*)\] -> \1
        ScalarType -> float

        # clean up missed std:: types
        std\.reference_wrapper\[([^\]]*)\] -> \1
        std\.span\[([^\]]*)\] -> List[\1]
        std\.set\[([^\]]*)\] -> Set[\1]

        # clean up invalid defaults
        =\s*Array\s*<[^>]+>\s*\(\) -> REMOVE
        =\s*Arr2D\s*<[^>]+>\s*\(\) -> REMOVE
        =\s*Vec\s*<[^>]+>\s*\(\) -> REMOVE
        =\s*Mat\s*<[^>]+>\s*\(\) -> REMOVE
    """.splitlines()
        if len(s := r.split("#")[0].strip()) > 0
    ]

    for pattern, replacement in patterns:
        if replacement == "REMOVE":
            replacement = ""
        regex = re.compile(pattern)
        while True:
            print(pattern, len(regex.findall(code)))
            new_code = regex.sub(replacement, code)
            if new_code == code:
                break
            code = new_code

    # map <submodule>.<type> to just <type> within submodules
    submodule_names = re.compile(r"# <submodule ([^>]+)>")
    for submodule in submodule_names.findall(code):
        full_submodule = re.compile(
            rf"# <submodule {submodule}>[\s\S]*# <\/submodule {submodule}>"
        ).search(code)
        assert full_submodule is not None
        full_submodule = full_submodule.group(0)
        fixed_submodule = full_submodule

        # fix types for each class defined in the submodule
        class_names = re.compile(r"class\s(\w[\w\d]+)(?:\([^\)]*\))?:")
        for name in class_names.findall(full_submodule):
            if name == submodule:
                continue

            # replace overly qualified references to the type
            regex = re.compile(rf"{submodule}\.{name}")
            fixed_submodule = regex.sub(name, fixed_submodule)

            # replace default values that do not resolve correctly
            default_regex = re.compile(rf"=\s{name}\(\)")
            fixed_submodule = default_regex.sub("", fixed_submodule)

        code = code.replace(full_submodule, fixed_submodule)

    return code


def amalgamate():
    options = amalgamated_header.AmalgamationOptions()

    options.base_dir = str(INCLUDE_DIR)
    options.local_includes_startwith = (
        "goblin/"  # Only include headers starting with "goblin"
    )
    options.include_subdirs = [
        "goblin",
        "goblin/lib",
        "goblin/gp",
        "goblin/bench",
        "goblin/methods",
    ]  # only include headers in these directories
    options.main_header_file = "goblin.h"  # The main header file
    options.dst_amalgamated_header_file = str(
        PYLIB_DIR / "include" / "amalgamation.h"
    )  # The destination file

    amalgamated_header.write_amalgamate_header_file(options)


def configure_litgen() -> litgen.LitgenOptions:
    # configure your options here
    options = litgen.LitgenOptions()

    # the C++ already conforms to the Python convention of
    # PascalCase for classes and snake_case for methods/functions
    # options.python_convert_to_snake_case = False

    # options.bind_library = litgen.BindLibraryType.nanobind
    options.use_nanobind()

    # the root namespace, no submodule will be generated for this
    options.namespaces_root = ["goblin"]

    # some replacements to clean up the definitions made in goblin/lib/types.h
    options.type_replacements = type_replacements()
    options.postprocess_stub_function = postprocess_stub

    # options.class_exclude_by_name__regex = "Instance|Problem"
    # options.class_template_options.add_specialization(
    #     "^SoASet$|^SolutionHandle$",
    #     ["Eigen::ColMajor", "Eigen::RowMajor"],
    #     cpp_synonyms_list_str=["ColMajor=Eigen::ColMajor", "RowMajor=Eigen::RowMajor"],
    # )
    # options.fn_return_force_policy_reference_for_references__regex = "^Objectives$"

    # Allow overriding all *Base classes in Python
    # - if this is not enabled, it can be the case that bindings
    # for the non-existent default constructor are generated
    options.class_override_virtual_methods_in_python__regex = "Base$"
    options.member_exclude_by_name__regex = "^_private|^request_debug_report$|^log_"

    # For reasons (I do not fully understand) the 'quality' Solution(Base) members get bound as
    #
    # ```
    # ...
    # .def("quality",
    #     [](goblin::Solution & self) { return self.quality(); })
    # .def("quality",
    #     [](goblin::Solution & self) { return self.quality(); })
    # ...
    # ```
    #
    # instead of (an overload without lambdas and explicit reference return value policy)
    #
    # ```
    # ...
    # .def("quality", nb::overload_cast<>(&goblin::Solution::quality, nb::const_),
    #     nb::rv_policy::reference_internal)
    # .def("quality", nb::overload_cast<>(&goblin::Solution::quality),
    #     nb::rv_policy::reference_internal)
    # ...
    # ```
    #
    # causing:
    #
    # ```
    # goblin/pylib/nanobind/bindings.cpp:1436:52: error:
    # allocating an object of abstract class type 'QualityBase'
    #  1436 |           [](goblin::SolutionBase & self) { return self.quality(); },
    #       |                                                    ^
    # goblin/lib/fitness.h:28:39: note:
    # unimplemented pure virtual method 'clone' in 'QualityBase'
    #    28 |  virtual std::unique_ptr<QualityBase> clone() const = 0;
    #       |                                       ^
    # goblin/pylib/nanobind/bindings.cpp:1440:52: error:
    # allocating an object of abstract class type 'QualityBase'
    #  1440 |           [](goblin::SolutionBase & self) { return self.quality(); },
    #       |                                                    ^
    # goblin/pylib/nanobind/bindings.cpp:1492:48: error:
    # allocating an object of abstract class type 'QualityBase'
    #  1492 |           [](goblin::Solution & self) { return self.quality(); },
    #       |                                                ^
    # goblin/pylib/nanobind/bindings.cpp:1496:48: error:
    # allocating an object of abstract class type 'QualityBase'
    #  1496 |           [](goblin::Solution & self) { return self.quality(); },
    #       |                                                ^
    # 4 errors generated.
    # ```
    #
    # So the fix (for now) is to explicitly NOT generate bindings
    # and manually add bindings returning qualities as references
    options.member_exclude_by_name_and_class__regex = {
        "Solution": r"quality",
        "SolutionBase": r"quality",
        "PSOState": r"quality",
    }
    for cls in ["goblin::SolutionBase", "goblin::Solution"]:
        options.custom_bindings.add_custom_bindings_to_class(
            qualified_class=cls,
            stub_code="""
                def quality(self) -> QualityBase:
                    ...
            """,
            pydef_code=f"""
                LG_CLASS.def("quality", nb::overload_cast<>(&{cls}::quality, nb::const_), nb::rv_policy::reference_internal);
                LG_CLASS.def("quality", nb::overload_cast<>(&{cls}::quality), nb::rv_policy::reference_internal);
            """,
        )
    for cls in ["goblin::classic::PSOState"]:
        options.custom_bindings.add_custom_bindings_to_class(
            qualified_class=cls,
            stub_code="""
                def previous_best_quality(self) -> QualityBase:
                    ...
            """,
            pydef_code=f"""
                LG_CLASS.def("previous_best_quality", nb::overload_cast<>(&{cls}::previous_best_quality, nb::const_), nb::rv_policy::reference_internal);
                LG_CLASS.def("previous_best_quality", nb::overload_cast<>(&{cls}::previous_best_quality), nb::rv_policy::reference_internal);
            """,
        )

    # These functions are excluded
    # - format_as: messes with the bindings and not needed on the Python side
    options.fn_exclude_by_name__regex = (
        r"^format_as$|^request_debug_report$|^seeded_rng$"
    )

    # Format the python stubs with black
    options.python_run_black_formatter = True

    return options


def main() -> None:
    # 1. amalgamate all headers into one
    # - fixes an issue where base classes defined in other headers
    #   are not registered as such (https://github.com/pthom/litgen/issues/23)
    amalgamate()

    # 2. generate the nanobind binding code
    litgen.write_generated_code_for_files(
        options=configure_litgen(),
        input_cpp_header_files=[str(PYLIB_DIR / "include" / "amalgamation.h")],
        output_cpp_pydef_file=REPO_DIR / "pylib" / "nanobind" / "bindings.cpp",
        output_stub_pyi_file=REPO_DIR / "pylib" / "pygom" / "__init__.pyi",
    )


if __name__ == "__main__":
    main()
