'''
========
[INTENT]
========
The main purpose of this program is to detect the tree structure in cricket
rulesets and output the tree structure to a .yaml file. A secondary purpose
is to write empty rules to a .txt file, since empty rules are a likely indication
of an OCR scanning error. The user can then manually correct these errors.
'''

# ==========================
#        [0] INPUTS        
# NOTE: REQUIRES USER INPUT!
# ==========================
# Fill in a year label
desired_year_label = "1947"

# ===========
# [1] IMPORTS
# ===========

import pandas as pd
import os
import re
import roman

# =======================================
# [2] SET-UP FOR DIRECTORY AND FILE PATHS
# =======================================

# -- [2.1] Maps a chosen year to the corresponding file prefix --
# E.g. "2019" gets mapped to "2019-laws-of-cricket-SY-tree", the folder name of
# the folder containing the .txt file with 2019's rules. We also call this the
# tree_folder. This step exists to remove the hassle of manually writing
# out the full file path each time you change years. 

df_year_lab_to_file_prefix = pd.read_excel('/Users/Chia Jia Nuo Daniel Personal Folder/After Touchdown/Santa Fe Institute/Rules Project Materials/Code and Data/Cricket/Year Label to File Prefix.xlsx',
                                        index_col = "Year Label")
df_year_lab_to_file_prefix.index = df_year_lab_to_file_prefix.index.astype(str)
tree_folder = df_year_lab_to_file_prefix.loc[desired_year_label, "Tree Folder"]

# From tree folder name, get the prefix of the cleaned .txt file
cleaned_file_name = tree_folder.removesuffix('tree') + 'cleaned-gemini'

# Change working directory
os.chdir('/Users/Chia Jia Nuo Daniel Personal Folder/After Touchdown/Santa Fe Institute/Rules Project Materials/Code and Data/Cricket/Rule Trees/Rule Tree Data')

# Initialize .yaml output file. This will hold the ruleset tree structure.
yaml_file_contents = '---\n(full ruleset):\n'

# ====================
# [3] HELPER FUNCTIONS
# ====================

# -- [3.2] Paragraph detection for lines --

def ends_with_period(line): 
    stripped_line = line.strip()
    # if stripped_line.endswith("."): 
    if stripped_line.endswith(".") or stripped_line.endswith('."') or stripped_line.endswith('.”'): 
        return True
    else: 
        return False

# -- [3.3] Writing into .yaml file --

# After detecting a new rule decimal, write in paras from the previous rule
# into tree, based on current tree depth. 
def write_in_new_paras(curr_depth, num_paras, existing_str, is_child, was_notes): 

    if was_notes: curr_depth += 1

    # Initialize string
    addition = ''

    # Iterate over num_paras, adding a new line entry for each para
    for i in range(num_paras): 
        # Preceding whitespace
        new_para_entry = 2 * (curr_depth+1) * ' ' # = 2 * (depth+1) * single space
        # Actual para entry with new line
        new_para_entry = new_para_entry + f'- P{i+1}\n'
        # Tack-on each new para entry
        addition = addition + new_para_entry

    # Result to return
    new_str = existing_str + addition

    # If the previous rule has no paragraphs detected, but the current rule
    # is NOT a child of it: add in a paragraph to the previous rule anyway,
    # since no childless rule should be empty. 
    if (num_paras == 0 and not is_child): 
        new_str = new_str + 2 * (curr_depth+1) * ' ' + f'- P1\n'

    return new_str

def new_little_law(existing_str, rule, curr_depth, is_notes): 
    if rule.endswith('.—'): 
        rule = rule.removesuffix('.—')
    if rule.endswith('.-'): 
        rule = rule.removesuffix('.-')
    if is_notes:
        curr_depth += 1
        new_line = (2 * curr_depth * ' ') + '- "N' + rule.rstrip() + '":\n'
    else: 
        new_line = (2 * curr_depth * ' ') + '- "' + rule.rstrip() + '":\n'
    result = existing_str + new_line
    return result

# -- [3.5] Regex --

# Takes in a list of entries and outputs a pattern to match. 
def get_pattern(candidates): 
    return (re.compile('|'.join(map(re.escape, candidates))))

# -- [3.6] Conversion from index to enumeration -- 

def get_num_ws(idx): 
    return f"{idx}. " # Note whitespace

def get_num_nl(idx): 
    return f"{idx}.\n" # Note newline

def get_letter(idx): 
    ascii_val = idx + 96 # e.g. 'a' is 1st alphabet, ascii value 97
    result = f"({chr(ascii_val)})"
    return result # note no whitespace at the end

def get_roman(idx): 
    return (f"({roman.toRoman(idx)})").lower()

def get_enum_type(string): 
    if (string[0]).isdigit(): 
        return 'law'
    elif (string[1]).isalpha() and 97 <= ord(string[1]) <= 104: 
        return 'letter'
    elif (string[1]).isalpha() and 104 < ord(string[1]): # only has i, v and x
                                                         # in first digit position
        return 'roman'
    else: 
        return 'ERROR!'

def strip_prefix(string): 
    if string.startswith('and '): 
        string = string.removeprefix('and ')
    elif string.startswith('or '): 
        string = string.removeprefix('or ')
    elif string.startswith('either '): 
        string = string.removeprefix('either ')
    return string

# =================
# [4] MAIN FUNCTION
# =================



# ============
# [MY SANDBOX]
# ============

with open(f"{tree_folder}/{cleaned_file_name}.txt", 'r') as f:

    # Initialize some variables
    curr_law_num = 1
    para_counter = 0
    is_notes = False
    was_notes = False
    is_empty = True # is the body of the rule we're in empty so far?
    
    problematic_lines_to_flag = '' # we flag rules that have no text, barring
                                # whitespace, as those signify scanning errors


    enum = '(A)'
    enum_type = 'header' # won't matter, just for initialization
    depth_dict = {'header': 1, 'law': 2, 'roman': 3} # for reg, not for notes
    enum_type_depth = depth_dict[f'{enum_type}']

    # A ballsy move. *No candidates.* A look at the document seems to suggest that
    # without tracking numbers, we can have "#. ", (letter) and (roman) and it will suffice.
    # (Also no (h), thankfully.)

    # Just use the regex pattern
    pattern = re.compile(r'^(?:\d+\.\—|\d+\.\-|\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\))')
    header_pattern = re.compile(r'^\([A-Z]\)')
    caps_pattern = re.compile(r'^[A-Z\s\.]+\.?$')

    # Loop over every line
    for line in f: 
        line = line.replace('\f', '')  # Remove form feed character

        # Check first for "(A), (B) etc". No need for space. 
        header_pattern_match = header_pattern.search(line)
        if header_pattern_match:
            # law_num = extract_law_num(line) # extract law_num
            new_enum = header_pattern_match.group()
            new_enum_type = 'header'
            new_enum_type_depth = depth_dict[f'{new_enum_type}']

            # para adding
            yaml_file_contents = write_in_new_paras(enum_type_depth, para_counter, yaml_file_contents, True, was_notes)

            is_notes = False
            was_notes = is_notes
            para_counter = 0
            yaml_file_contents = yaml_file_contents + f'  - {new_enum}:\n' # make node

            enum = new_enum
            enum_type = new_enum_type
            enum_type_depth = new_enum_type_depth


        # Now, detect law, roman
        match = pattern.search(line)
        if match: # auto-detects at start
            new_enum = match.group()
            new_enum_type = get_enum_type(new_enum)
            new_enum_type_depth = depth_dict[f'{new_enum_type}']

            if new_enum: 
                is_child = new_enum_type_depth > enum_type_depth
                yaml_file_contents = write_in_new_paras(enum_type_depth, para_counter, yaml_file_contents, is_child, was_notes)
                
                if new_enum.endswith('.—'):
                    if new_enum_type == 'law' and int(new_enum.removesuffix('.—')) == curr_law_num: 
                        is_notes = False
                        curr_law_num += 1
                elif new_enum.endswith('.-'):
                    if new_enum_type == 'law' and int(new_enum.removesuffix('.-')) == curr_law_num: 
                        is_notes = False
                        curr_law_num += 1
                # NEW LAW
                yaml_file_contents = new_little_law(yaml_file_contents, new_enum, new_enum_type_depth, is_notes)
          
            para_counter = 0
            enum = new_enum
            enum_type = new_enum_type
            enum_type_depth = new_enum_type_depth
            was_notes = is_notes # only set this if there is an enumeration match

        caps_match = caps_pattern.search(line)
        if caps_match and line != '\n': 
            is_notes = False
            if desired_year_label == "1952" or desired_year_label == "1947":
                para_counter -= 1

        if line.startswith("NOTES") or line.startswith("NOTE"): is_notes = True

            # END OF FOR LOOP

        if ends_with_period(line): para_counter += 1
    
    # [(e) last steps after iterating over every line of document]
    # At the end of entire loop, when document is exhausted, add the final set of paras
    yaml_file_contents = write_in_new_paras(enum_type_depth, para_counter, yaml_file_contents, True, was_notes)

    # And print out the problematic lines to flag
    if problematic_lines_to_flag != '': # as long as not the empty string
        problematic_lines_to_flag = problematic_lines_to_flag.removesuffix(', ')

# Output: write the .yaml contents to a .yaml file!
with open(f"{tree_folder}/{desired_year_label}-tree-structure-raw.yaml", "w") as f: 
    f.write(yaml_file_contents)

# And write the "Problematic lines to flag" to a .txt file. Also printed.
with open(f"{tree_folder}/{desired_year_label}-problematic-lines.txt", "w") as f:
    f.write(f"Problematic lines to flag: {problematic_lines_to_flag}")


# ================
# [END OF SANDBOX]
# ================

print("STOP HERE! You have left sandbox. (1)")
print("STOP HERE! You have left sandbox. (2)")
print("STOP HERE! You have left sandbox. (3)")

