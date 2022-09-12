# include component requirements which were introduced after IDF version 4.1
if("${IDF_VERSION_MAJOR}.${IDF_VERSION_MINOR}" VERSION_GREATER_EQUAL "5.0")
    list(APPEND priv_req driver)
endif()


idf_component_register(SRC_DIRS "."
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES ${priv_req})
