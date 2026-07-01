#pragma once

#ifdef DEBUG
    #define DEBUG_ONLY(stmt) \
        do {                 \
            stmt;            \
        } while (0)
#else
    #define DEBUG_ONLY(stmt) \
        do {                 \
        } while (0)
#endif
