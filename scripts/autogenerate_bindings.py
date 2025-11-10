import os
import pathlib

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
    \bstd::make_shared<(.*)>\b -> \1
    \bstd::filesystem::path\b -> pathlib.Path | str
    \bstd::chrono::nanoseconds\b -> datetime.timedelta
    \bstd::span<(.*)>\b -> List[\1]
    \bstd::chrono::seconds\((\d+)\)\b -> datetime.timedelta(seconds=\1)
    \bstd::chrono::minutes\((\d+)\)\b -> datetime.timedelta(minutes=\1)

    \bVec<(.*)>\b -> np.ndarray
    \bArray<(.*)>\b -> np.ndarray
    \bMat<(.*)>\b -> np.ndarray
    \bArr2D<(.*)>\b -> np.ndarray
    \bActive\b -> np.ndarray
    \bC?RefS?<(.*)>\b -> np.ndarray
    """

    replaces = litgen.RegexReplacementList.from_string(
        replacements_str + customm_replacements_str
    )
    return replaces


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

    options.bind_library = litgen.BindLibraryType.nanobind

    # the root namespace, no submodule will be generated for this
    options.namespaces_root = ["goblin"]

    # some replacements to clean up the definitions made in goblin/lib/types.h
    options.type_replacements = type_replacements()
    options.type_replacements.add_last_replacement(
        r"MOFitness.Quality", r'"MOFitness.Quality"'
    )

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

    # These functions are excluded
    # - format_as: messes with the bindings and not needed on the Python side
    options.fn_exclude_by_name__regex = r"^format_as$|^request_debug_report$"

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
