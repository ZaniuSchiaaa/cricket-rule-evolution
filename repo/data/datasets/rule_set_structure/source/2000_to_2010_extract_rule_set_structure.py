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
desired_year_label = "2000"
# Fill in titles that doc contains, e.g. ["THE PREAMBLE", "APPENDIX A", "APPENDIX B"]
worded_titles = ["THE PREAMBLE", "APPENDIX D"]


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

def starts_with_bullet(line): 
    if line.startswith("•"): 
    # if line.startswith("■") or line.startswith("□"): 
        return True
    else: 
        return False

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
def write_in_new_paras(curr_depth, num_paras, existing_str, is_child): 

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

# Add a new law that is not of the above big law form. 
# E.g. "17.2", "17.5.1". 
def new_little_law(existing_str, rule, curr_depth): 
    new_line = (2 * curr_depth * ' ') + '- "' + rule.rstrip() + '":\n'
    result = existing_str + new_line
    return result

# Add a title that's not a decimal, e.g. "THE PREAMBLE"
def new_title(existing_str, title): 
    result = existing_str + '  - ' + title + ':\n'
    return result

# -- [3.4] Add whitespace to strings --

# Function to add a space at the end of a string
def add_space(string):
    return string + ' '

def add_newline(string):
    return string + '\n'

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
        return 'num'
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

# Open the .txt file with raw ruleset
with open(f"{tree_folder}/{cleaned_file_name}.txt", 'r') as f:

    # [0] Initialize variables
    depth = 0 # now, this is adjusted by number / letter / roman
    para_counter = 0
    in_bullet = False # are we currently in a paragraph tagged to a bullet point?
    is_empty = True # is the body of the rule we're in empty so far?
    enum_type = None

    # another lengthy reset...for preamble.
    curr_law_num = 0
    target_law_num = 1
    
    worded_titles_pattern = get_pattern(worded_titles)

    curr_num_idx = None
    target_num_idx = 1
    target_num_ws = get_num_ws(target_num_idx)
    target_num_nl = get_num_nl(target_num_idx)

    curr_letter_idx = None
    target_letter_idx = 1
    target_letter = get_letter(target_letter_idx)

    curr_roman_idx = None
    target_roman_idx = 1
    target_roman = get_roman(target_roman_idx)

    candidates = [target_num_ws, target_num_nl, target_letter, target_roman]
    pattern = get_pattern(candidates)

    # Misc
    problematic_lines_to_flag = '' # we flag rules that have no text, barring
                                # whitespace, as those signify scanning errors

    enum_type_depth = {'num': 2, 'letter': 3, 'roman': 4}

    for line in f: 
        line = line.replace('\f', '')  # Remove form feed character
        line = strip_prefix(line)

        # [0] check for match with worded titles
        worded_titles_match = worded_titles_pattern.search(line) # gives bool
        if worded_titles_match and line.startswith(worded_titles_match.group()): 
            # (a)(i) Actions if worded title is found
            # Write in new paragraphs from previous rule FIRST
            yaml_file_contents = write_in_new_paras(depth, para_counter, yaml_file_contents, True)
            # After writing in paragraphs, reset paragraph count
            para_counter = 0
            # Then, add the worded title into .yaml contents
            yaml_file_contents = new_title(yaml_file_contents, title=worded_titles_match.group())
            # Set depth to 1
            depth = 1
            # We are no longer in bullet, hence: 
            in_bullet = False

        # [1] Found a new law
        if line.startswith(f"LAW {target_law_num}"): 
            # Write in new paragraphs from previous rule FIRST
            yaml_file_contents = write_in_new_paras(depth, para_counter, yaml_file_contents, False)
            # After writing in paragraphs, reset paragraph count
            para_counter = 0

            # Write in new law
            yaml_file_contents = yaml_file_contents + f'  - LAW {target_law_num}: \n'

            curr_law_num = target_law_num
            target_law_num += 1 # updating search for law

            # hardcoded reset (might change later)
            curr_num_idx = None
            target_num_idx = 1
            target_num_ws = get_num_ws(target_num_idx)
            target_num_nl = get_num_nl(target_num_idx)

            curr_letter_idx = None
            target_letter_idx = 1
            target_letter = get_letter(target_letter_idx)

            curr_roman_idx = None
            target_roman_idx = 1
            target_roman = get_roman(target_roman_idx)

            candidates = [target_num_ws, target_num_nl, target_letter, target_roman]
            pattern = get_pattern(candidates)

            depth = 1
            in_bullet = False
            enum_type = None

        # [2] Found a new enumeration
        if pattern: 
            match = pattern.search(line)
        # sorry for the jank None checking
        if pattern and match and line.startswith(match.group()):
            new_enum_type = get_enum_type(match.group())
            # (b)(i) Write in new paragraphs. 
            # But first, a bool that tells you if newly found rule is child of previous rule.
            
            is_child = enum_type is None or \
                enum_type_depth[f'{new_enum_type}'] > enum_type_depth[f'{enum_type}']
            yaml_file_contents = write_in_new_paras(depth, para_counter, yaml_file_contents, is_child)
            # After writing in paragraphs, reset the paragraph count
            para_counter = 0
            
            # I'm going to cheat a little, because from 2000 to 2010, the enumeration
            # of letters does not go beyond (g). So I just need to check the ascii. 
            enum_type = new_enum_type

            # Update problematic_lines_to_flag if previous rule was empty.
            # Sequence has been CHANGED so that we can have updated indexes. 
            if is_empty == True: 
                problematic_lines_to_flag = problematic_lines_to_flag + \
                    f'{curr_law_num}|{curr_num_idx}|{curr_letter_idx}|{curr_roman_idx}' + ', '  

            # Reset is_empty, and then see if current line has text apart from
            # the rule enumeration itself
            is_empty = True
            if (line.removeprefix(match.group())).strip() != '':
                is_empty = False

            if enum_type == 'num': 
                yaml_file_contents = new_little_law(yaml_file_contents, match.group(), enum_type_depth[f'{enum_type}'])
                curr_num_idx = target_num_idx
                target_num_idx += 1 # increment
                target_num_ws = get_num_ws(target_num_idx)
                target_num_nl = get_num_nl(target_num_idx)
                # also, reset deeper layers
                curr_letter_idx = None
                target_letter_idx = 1 # reset
                target_letter = get_letter(target_letter_idx)

                curr_roman_idx = None
                target_roman_idx = 1 # reset
                target_roman = get_roman(target_roman_idx)
      
            elif enum_type == 'letter': 
                yaml_file_contents = new_little_law(yaml_file_contents, match.group(), enum_type_depth[f'{enum_type}'])
                curr_letter_idx = target_letter_idx
                target_letter_idx += 1 # increment
                target_letter = get_letter(target_letter_idx)

                # also, reset deeper layers
                curr_roman_idx = None
                target_roman_idx = 1 # reset
                target_roman = get_roman(target_roman_idx)

                candidates = [target_num_ws, target_num_nl, target_letter, target_roman]
                pattern = get_pattern(candidates)

                # MAYBE HAVE A SUB SEARCH HERE. For if there is a space there. 
                line = (line.removeprefix(match.group())).lstrip()
                if pattern: 
                    sub_search_match = pattern.search(line)
                    if pattern and sub_search_match and line.startswith(sub_search_match.group()): 
                        match = sub_search_match
                        new_enum_type = get_enum_type(match.group())
                        enum_type = new_enum_type
                        if enum_type == 'roman':
                            yaml_file_contents = new_little_law(yaml_file_contents, match.group(), enum_type_depth[f'{enum_type}'])
                            curr_roman_idx = target_roman_idx
                            target_roman_idx += 1
                            target_roman = get_roman(target_roman_idx)

            elif enum_type == 'roman':
                yaml_file_contents = new_little_law(yaml_file_contents, match.group(), enum_type_depth[f'{enum_type}'])
                curr_roman_idx = target_roman_idx
                target_roman_idx += 1
                target_roman = get_roman(target_roman_idx)

            else: 
                print('ERROR!') 
            
            # update candidates
            candidates = [target_num_ws, target_num_nl, target_letter, target_roman]
            pattern = get_pattern(candidates)

            # update depth
            depth = enum_type_depth[f'{enum_type}']

        # [3] else if no rule detected, just conditionally update is_empty
        else: 
            if(line.strip() != ''): is_empty = False 

        # [4] use paragraph detection logic to increment paragraph counter]
        if starts_with_bullet(line): 
            para_counter += 1
            in_bullet = True
        
        if ends_with_period(line): 
            if not in_bullet: 
                para_counter += 1
            else: # i.e. in_bullet
                in_bullet = False # reset
                # and do not increment paragraph counter, as this line belonged
                # to same "paragraph" as the previously counted bullet point

    # [(e) last steps after iterating over every line of document]
    # At the end of entire loop, when document is exhausted, add the final set of paras
    yaml_file_contents = write_in_new_paras(depth, para_counter, yaml_file_contents, True)

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

# with open(f"{tree_folder}/{cleaned_file_name}.txt", 'r') as f:


#     # -- [4.1] Initialize variables --

#     # For identifying rules and sections: 
#     candidates = ["1.1 ", "1.1\n"] # Initial candidates list, hardcoded for first instance
#     pattern = get_pattern(candidates) # pattern for candidates
#     worded_titles_pattern = get_pattern(worded_titles) # pattern for worded titles,
#                                                        # see section [0] above

#     # State variables                                                   
#     depth = 0 # tree depth
#     para_counter = 0 # counts num of paragraphs
#     curr_rule = None # what rule are we looking at now?
#     in_bullet = False # are we currently in a paragraph tagged to a bullet point?
#     is_empty = True # is the body of the rule we're in empty so far?
    
#     # Misc
#     problematic_lines_to_flag = '' # we flag rules that have no text, barring
#                                 # whitespace, as those signify scanning errors

#     # -- [4.2] Loop over EVERY line sequentially --
#     for line in f: 
#         line = line.replace('\f', '')  # Remove form feed character

#         # [(a) check for match with worded titles]
#         worded_titles_match = worded_titles_pattern.search(line) # gives bool
#         if worded_titles_match and line.startswith(worded_titles_match.group()): 
#             # (a)(i) Actions if worded title is found
#             # Write in new paragraphs from previous rule FIRST
#             yaml_file_contents = write_in_new_paras(depth, para_counter, yaml_file_contents, True)
#             # After writing in paragraphs, reset paragraph count
#             para_counter = 0
#             # Then, add the worded title into .yaml contents
#             yaml_file_contents = new_title(yaml_file_contents, title=worded_titles_match.group())
#             # Set depth to 1
#             depth = 1
#             # We are no longer in bullet, hence: 
#             in_bullet = False
            
#             # (a)(ii) Check if title is an appendix. If, so update candidates.
#             for elem in worded_titles: 
#                 if elem.startswith("APPENDIX"): 
#                     if worded_titles_match.group() == elem:
#                         last_letter = elem[-1]
#                         candidates = [f"{last_letter}.1 ", f"{last_letter}.1\n"]
#                         pattern = get_pattern(candidates)
#                         curr_rule = elem # for checking in should_insert_filler_para later

#         #  [(b) Check for match with candidates — e.g. 3.1, A.3.2]
#         match = pattern.search(line) # returns bool
#         if match and line.startswith(match.group()):            
#             # (b)(i) Write in new paragraphs. 
#             # But first, a bool that tells you if newly found rule is child of previous rule.
#             is_child = (curr_rule is not None) and (rule_len(match.group()) > rule_len(curr_rule))
#             yaml_file_contents = write_in_new_paras(depth, para_counter, yaml_file_contents, is_child)
#             # After writing in paragraphs, reset the paragraph count
#             para_counter = 0

#             # (b)(ii) Update problematic_lines_to_flag if previous rule was empty.
#             if is_empty == True: 
#                 problematic_lines_to_flag = problematic_lines_to_flag + curr_rule.strip() + ', '

#             # (b)(iii) Update state variables
#             curr_rule = match.group() # i.e. update curr_rule to what term was matched
#             depth = rule_len(curr_rule) # update current depth
#             in_bullet = False # we are no longer in a bullet point
            
#             # Get new rules to search for, based on updated rule
#             candidates = generate_next_candidates(curr_rule)
#             pattern = get_pattern(candidates) 

#             # Reset is_empty, and then see if current line has text apart from
#             # the rule decimal itself
#             is_empty = True
#             if (line.removeprefix(match.group())).strip() != '':
#                 is_empty = False

#             # (b)(iv) Create rules in .yaml string.
#             # If of the form a.1, create two layer law headers (Law a --> a.1)
            
#             # Note: we choose to detect laws of the form "a.1" rather than of the
#             # form "Law a", because some of the latter were lost in the OCR. 
#             # Nevertheless, we still plot both layers in. 
#             if rule_len(curr_rule) == 2 and parse_rule(curr_rule)[-1] == 1 \
#                 and parse_rule(curr_rule)[0].isdigit():
#                 yaml_file_contents = new_big_law(yaml_file_contents, match.group())
            
#             # If any other kind of law is detected, one layer of law headers will do
#             else: 
#                 yaml_file_contents = new_little_law(yaml_file_contents, match.group(), depth)

#         # [(c) else if no rule detected, just conditionally update is_empty]
#         else: 
#             if(line.strip() != ''): is_empty = False 

#         # [(d) use paragraph detection logic to increment paragraph counter]
#         if starts_with_bullet(line): 
#             para_counter += 1
#             in_bullet = True
        
#         if ends_with_period(line): 
#             if not in_bullet: 
#                 para_counter += 1
#             else: # i.e. in_bullet
#                 in_bullet = False # reset
#                 # and do not increment paragraph counter, as this line belonged
#                 # to same "paragraph" as the previously counted bullet point

#     # [(e) last steps after iterating over every line of document]
#     # At the end of entire loop, when document is exhausted, add the final set of paras
#     yaml_file_contents = write_in_new_paras(depth, para_counter, yaml_file_contents, True)

#     # And print out the problematic lines to flag
#     if problematic_lines_to_flag != '': # as long as not the empty string
#         problematic_lines_to_flag = problematic_lines_to_flag.removesuffix(', ')

# # Output: write the .yaml contents to a .yaml file!
# with open(f"{tree_folder}/{desired_year_label}-tree-structure-raw.yaml", "w") as f: 
#     f.write(yaml_file_contents)

# # And write the "Problematic lines to flag" to a .txt file. Also printed.
# with open(f"{tree_folder}/{desired_year_label}-problematic-lines.txt", "w") as f:
#     f.write(f"Problematic lines to flag: {problematic_lines_to_flag}")

