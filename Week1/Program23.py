# Write a Python program to find the available built-in modules

# print (help('modules') )

import sys
import textwrap
#return built-in modules
module_name = ', '.join(sorted(sys.builtin_module_names))
print(textwrap.wrap(module_name, width=70))
