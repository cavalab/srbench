//#include <limits>
#include <sstream>
//#include <utility>
#include <vector>

#include <bingocpp/agraph/string_generation.h>
#include <bingocpp/agraph/operator_definitions.h>
#include <bingocpp/agraph/constants.h>

namespace bingo {
namespace string_generation {
namespace {

std::string get_formatted_element_string(const Eigen::ArrayX3i &stack_element,
                                         std::vector<std::string> string_list,
                                         const PrintMap &format_map,
                                         const Eigen::VectorXd &constants);

std::string get_stack_string(const Eigen::ArrayX3i &command_array,
                             const Eigen::VectorXd &constants);

std::string get_stack_element_string(const Eigen::VectorXd &constants,
                                     int command_index,
                                     const Eigen::ArrayX3i &stack_element);

std::string print_string_with_args(const std::string &string,
                                   const std::string &arg1,
                                   const std::string &arg2);
} // namespace



std::string GetFormattedString(const std::string format,
                               const Eigen::ArrayX3i &command_array,
                               const Eigen::VectorXd &constants){
  if (format.compare("stack") == 0) {
      return get_stack_string(command_array, constants);
  }

  PrintMap format_map;
  if (format.compare("latex") == 0) {
      format_map = kLatexPrintMap;
  } else {
      format_map = kConsolePrintMap;
  }

  std::vector<std::string> string_list;
  for (auto stack_element : command_array.rowwise()) {
    std::string temp_string = get_formatted_element_string(
        stack_element, string_list, format_map, constants);
    string_list.push_back(temp_string);
  }
  return string_list.back();
}

namespace {

std::string get_formatted_element_string(const Eigen::ArrayX3i &stack_element,
                                         std::vector<std::string> string_list,
                                         const PrintMap &format_map,
                                         const Eigen::VectorXd &constants) {
  int node = stack_element(0, kOpIdx);
  int param1 = stack_element(0, kParam1Idx);
  int param2 = stack_element(0, kParam2Idx);

  std::string temp_string;
  if (node == Op::kVariable) {
    temp_string = "X_" + std::to_string(param1);
  } else if (node == Op::kConstant) {
    if (param1 == kOptimizeConstant ||
        param1 >= constants.size()) {
      temp_string = "?";
    } else {
      temp_string = std::to_string(constants[param1]);
    }
  } else if (node == Op::kInteger) {
    temp_string = std::to_string(param1);
  } else {
    temp_string = print_string_with_args(format_map.at(node),
                                         string_list[param1],
                                         string_list[param2]);
  }
  return temp_string;
}

std::string print_string_with_args(const std::string &string,
                                   const std::string &arg1,
                                   const std::string &arg2) {
  std::stringstream stream;
  bool first_found = false;
  for (std::string::const_iterator character = string.begin();
       character != string.end(); character++) {
    if (*character == '{' && *(character + 1) == '}') {
      stream << ((!first_found) ? arg1 : arg2);
      character++;
      first_found = true;
    } else {
      stream << *character;
    }
  }
  return stream.str();
}

std::string get_stack_string(
    const Eigen::ArrayX3i &command_array,
    const Eigen::VectorXd &constants) {
  std::string temp_string;
  for (int i = 0; i < command_array.rows(); i++) {
    temp_string += get_stack_element_string(constants, i, command_array.row(i));
  }
  return temp_string;
}

std::string get_stack_element_string(const Eigen::VectorXd &constants,
                                     int command_index,
                                     const Eigen::ArrayX3i &stack_element) {
  int node = stack_element(0, kOpIdx);
  int param1 = stack_element(0, kParam1Idx);
  int param2 = stack_element(0, kParam2Idx);

  std::string temp_string = "("+ std::to_string(command_index) +") <= ";
  if (node == Op::kVariable) {
    temp_string += "X_" + std::to_string(param1);
  } else if (node == Op::kConstant) {
    if (param1 == kOptimizeConstant ||
        param1 >= constants.size()) {
      temp_string += "C";
    } else {
      temp_string += "C_" + std::to_string(param1) + " = " + 
                     std::to_string(constants[param1]);
    }
  } else if (node == Op::kInteger) {
    temp_string += std::to_string(param1) + " (integer)";
  } else {
    std::string param1_str = std::to_string(param1);
    std::string param2_str = std::to_string(param2);
    temp_string += print_string_with_args(kStackPrintMap.at(node),
                                          param1_str,
                                          param2_str);
  }
  temp_string += '\n';
  return temp_string;
}

} // namespace (anonymous)
} // namespace string_generation
} // namespace bingo