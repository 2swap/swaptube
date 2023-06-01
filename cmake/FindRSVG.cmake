# - Try to find libRSVG
# RSVG_FOUND - system has pango
# RSVG_LIBRARIES - path to the library
# RSVG_INCLUDE_DIRS - include dirs
# RSVG_DEFINITIONS
# 
# COPYING-CMAKE-SCRIPTS
# Alexander Couzens <lynxis@base45.de>

find_package(PkgConfig)
find_package(Cairo REQUIRED)

pkg_check_modules(PC_RSVG QUIET librsvg-2.0)
set(libRSVG_DEFINITIONS ${PC_RSVG_CFLAGS_OTHER})

find_path(RSVG_INCLUDE_DIR librsvg/rsvg.h
          HINTS ${PC_RSVG_INCLUDEDIR} ${PC_RSVG_INCLUDE_DIRS}
          )

find_library(RSVG_LIBRARY NAMES rsvg-2
             HINTS ${PC_RSVG_LIBDIR} ${PC_RSVG_LIBRARY_DIRS} )

set(RSVG_LIBRARIES ${RSVG_LIBRARY} ${CAIRO_LIBRARY})
set(RSVG_INCLUDE_DIRS ${RSVG_INCLUDE_DIR} ${CAIRO_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RSVG_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(RSVG  DEFAULT_MSG
                                  RSVG_LIBRARY RSVG_INCLUDE_DIR)

mark_as_advanced(RSVG_INCLUDE_DIR RSVG_LIBRARY)

