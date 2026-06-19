# Three centuries of the Laws of Cricket reveal core principles of the
evolution of regulatory mechanisms

This repository contains all the data files that were used for our analysis, as well as the figures used in the manuscript. Where possible, the scripts to create these data files and figures have been included.

## Repository Structure
```
repo/ # all scripts to be ran from within this folder
├── figures/          
│   ├── main_body/          
│   │   ├── fig1/          # timeline with key plots  
│   │   │   ├── fig1_final.svg
│   │   │   ├── fig1_final.pdf
│   │   │   ├── components/
│   │   │   │   ├── fig1_num_laws_and_ave_depth.svg
│   │   │   │   └── fig1_num_words_and_num_leaves.svg
│   │   │   └── source/
│   │   │   │   ├── fig1_num_laws_and_ave_depth.py
│   │   │   │   └── fig1_num_words_and_num_leaves.py
│   │   ├── fig2/          # flowchart of data pipeline  
│   │   │   ├── fig2_final.svg
│   │   │   └── fig2_final.pdf
│   │   ├── fig3/          # text analysis summary figure
│   │   │   ├── fig3_final.svg
│   │   │   ├── fig3_final.pdf
│   │   │   ├── components/
│   │   │   │   ├── fig3_a_matches_over_time.svg
│   │   │   │   ├── fig3_b_num_words_over_matches.svg
│   │   │   │   ├── fig3_c_num_unique_words_over_matches.svg
│   │   │   │   ├── fig3_d_zm_exp_over_time.svg
│   │   │   │   ├── fig3_e_rank_freq.png
│   │   │   │   └── fig3_f_discretion_words_over_matches.svg
│   │   │   └── source/
│   │   │   │   ├── fig3_a_matches_over_time.py
│   │   │   │   ├── fig3_b_num_words_over_matches.py
│   │   │   │   ├── fig3_c_num_unique_words_over_matches.py
│   │   │   │   ├── fig3_d_zm_exp_over_time.py
│   │   │   │   ├── fig3_e_rank_freq.py
│   │   │   │   └── fig3_f_discretion_words_over_matches.py
│   │   ├── fig4/          # tree structure nomenclature
│   │   │   ├── fig4_final.png
│   │   │   └── fig4_final.pdf
│   │   ├── fig5/          # rule evolution timeline
│   │   │   ├── fig5_final.svg
│   │   │   └── fig5_final.pdf
│   │   ├── fig6/          # tree structure summary figure
│   │   │   ├── fig6_final.svg
│   │   │   └── fig6_final.pdf 
│   │   │   ├── components/
│   │   │   │   ├── fig6_a_2019_tree_structure.png
│   │   │   │   ├── fig6_b_nodes_and_leaves_over_matches.svg
│   │   │   │   ├── fig6_c_laws_over_time.svg
│   │   │   │   ├── fig6_d_ave_br_factor_over_matches.svg
│   │   │   │   ├── fig6_e_ave_depth_over_matches.svg
│   │   │   │   └── fig6_f_max_depth_over_matches.svg
│   │   │   └── source/
│   │   │   │   ├── fig6_a_2019_tree_structure.py
│   │   │   │   ├── fig6_b_nodes_and_leaves_over_matches.py
│   │   │   │   ├── fig6_c_laws_over_time.py
│   │   │   │   ├── fig6_d_ave_br_factor_over_matches.py
│   │   │   │   ├── fig6_e_ave_depth_over_matches.py
│   │   │   │   └── fig6_f_max_depth_over_matches.py
│   │   ├── fig7/         # evolution of interdependency network
│   │   │   ├── fig7_final.svg
│   │   │   └── fig7_final.pdf
│   │   │   ├── components/
│   │   │   │   ├── fig7_1918_in_out_degree_weight_distributions.svg
│   │   │   │   ├── fig7_1962_in_out_degree_weight_distributions.svg
│   │   │   │   └── fig7_2019_in_out_degree_weight_distributions.svg
│   │   │   └── source/
│   │   │   │   └── fig7_in_out_degree_weight_distributions.py
│   │   ├── fig8/ 		# timeline for best model fit for in-weight and out-weight
│   │   │   ├── fit_parameters.csv
│   │   │   ├── fig8_final.svg
│   │   │   ├── fig8_final.pdf
│   │   │   └── source/
│   │   │   │   └── fig8.py
│   │   ├── fig9/         # interdependency network summary figure  
│   │   │   ├── fig9_final.svg
│   │   │   └── fig9_final.pdf
│   │   │   ├── components/
│   │   │   │   ├── fig9_a_ave_weight_over_time.svg
│   │   │   │   ├── fig9_b_gini_over_time.svg
│   │   │   │   ├── fig9_c_reciprocity_over_time.svg
│   │   │   │   ├── fig9_d_clust_coeff_over_time.svg
│   │   │   │   ├── fig9_e_modularity_over_time.svg
│   │   │   │   └── fig9_f_num_communities_over_time.svg
│   │   │   └── source/
│   │   │   │   ├── fig9_a_ave_weight_over_time.py
│   │   │   │   ├── fig9_b_gini_over_time.py
│   │   │   │   ├── fig9_c_reciprocity_over_time.py
│   │   │   │   ├── fig9_d_clust_coeff_over_time.py
│   │   │   │   ├── fig9_e_modularity_over_time.py
│   │   │   │   └── fig9_f_num_communities_over_time.py
│   │   ├── fig10/         # eigenvector_centrality_top_five
│   │   │   ├── fig10_final.svg
│   │   │   ├── fig10_final.pdf
│   │   │   └── source/
│   │   │   │   └── fig10.py
│   │   ├── fig11/         # eigenvector_centrality_case_studies
│   │   │   ├── fig11_final.svg
│   │   │   └── fig11_final.pdf
│   │   │   └── source/
│   │   │   │   └── fig11.py
│   └── supplemental/          
│   │   ├── figS1/
│   │   │   ├── figS1_final.svg
│   │   │   └── figS1_final.pdf
│   │   │   ├── components/
│   │   │   │   ├── figS1_a_total_weight_over_time.svg
│   │   │   │   ├── figS1_b_num_edges_over_time.svg
│   │   │   │   ├── figS1_c_ave_weight_over_existing_edges_over_time.svg
│   │   │   │   └── figS1_d_ave_weight_over_all_edges_over_time.svg
│   │   │   └── source/
│   │   │   │   ├── figS1_a_total_weight_over_time.py
│   │   │   │   ├── figS1_b_num_edges_over_time.py
│   │   │   │   ├── figS1_c_ave_weight_over_existing_edges_over_time.py
│   │   │   │   └── figS1_d_ave_weight_over_all_edges_over_time.py
│   │   ├── figS2/
│   │   │   ├── figS2_final.svg
│   │   │   └── figS2_final.pdf
│   │   │   └── source/
│   │   │   │   └── figS2.py
│   │   ├── figS3/
│   │   │   ├── figS3_final.svg
│   │   │   └── figS3_final.pdf
│   │   │   ├── components/
│   │   │   │   ├── figS3_1918.svg
│   │   │   │   ├── figS3_1968.svg
│   │   │   │   ├── figS3_2019.svg
│   │   │   │   └── figS3_heatmap.svg
│   │   │   └── source/
│   │   │   │   ├── figS3_plot_year.py
│   │   │   │   └── figS3_heatmap.py
│   │   ├── figS4/
│   │   │   ├── figS4_final.svg
│   │   │   └── figS4_final.pdf
│   │   │   └── source/
│   │   │   │   └── figS4.py
│   │   ├── figS5/
│   │   │   ├── figS5_final.svg
│   │   │   └── figS5_final.pdf
│   │   │   └── source/
│   │   │   │   └── figS5.py
│   │   ├── figS6/
│   │   │   ├── figS6_final.png
│   │   │   └── source/
│   │   │   │   └── figS6.py
│   │   ├── figS7/
│   │   │   ├── figS7_final.png
│   │   │   └── source/
│   │   │   │   └── figS7.py
│   │   ├── figS8/
│   │   │   ├── figS8_final.svg
│   │   │   └── figS8_final.pdf
│   │   │   └── source/
│   │   │   │   └── figS8.py
│   │   └── figS9/
│   │   │   ├── figS9_final.svg
│   │   │   └── figS9_final.pdf
│   │   │   └── source/
│   │   │   │   └── figS9.py
│
├── data/
│   ├── datasets/
│   │   ├── interdependency_networks/
│   │   │   ├── citation_tables/
│   │   │   │   ├── [....csv files from all years…]
│   │   │   │   └── source/
│   │   │   │   │   ├── enumerated_text/
│   │   │   │   │   │   └── [...csv files from all years…]
│   │   │   │   │   └── extract_citations.py
│   │   │   ├── graph_files/
│   │   │   │   ├── gexf/
│   │   │   │   │   ├── multi_count/
│   │   │   │   │   │   └── [....gexf files from all years…]
│   │   │   │   │   └── single_count/
│   │   │   │   │   │   └── [....gexf files from all years…]
│   │   │   │   └── source/
│   │   │   │   │   ├── extract_interdependency_network.csv
│   │   │   │   │   └── README.md
│   │   ├──number_of_matches/
│   │   │   └── cumulative_matches_played.csv
│   │   ├──rule_set_structure/
│   │   │   ├── yaml_files/
│   │   │   │   ├── original/
│   │   │   │   │   └── [....yaml files…]
│   │   │   │   ├── flattened/
│   │   │   │   │   └── [....yaml files…]
│   │   │   └── source/
│   │   │   │   │   ├── 2017_to_2019_extract_rule_set_structure.py
│   │   │   │   │   └── README.md
│   │   ├── rule_texts/
│   │   │   ├── raw/
│   │   │   │   └── [....txt files from all years…]
│   │   │   └── processed/
│   │   │   │   ├── [....txt files from all years…]
│   │   │   │   └── data_appendix.xlsx
│   └── visualizations/
│   │   ├── rule_set_structure/
│   │   │   ├── tree_layout/
│   │   │   │   ├── tree_viz/
│   │   │   │   │   ├── flattened/
│   │   │   │   │   │   └── [....html files from all years…]
│   │   │   │   │   └── original/
│   │   │   │   │   │   └── [....html files from all years…]
│   │   │   │   └── source/
│   │   │   │   │   └── rule_set_tree_viz.py/
│   │   │   ├── network_layout/
│   │   │   │   ├── network_viz/
│   │   │   │   │   ├── original/
│   │   │   │   │   │   ├── pngs/
│   │   │   │   │   │   │   └── [...png files from all years…]
│   │   │   │   │   │   └── svgs/
│   │   │   │   │   │   │   └── [...svg files from all years…]
│   │   │   │   │   └── flattened/
│   │   │   │   │   │   ├── pngs/
│   │   │   │   │   │   │   └── [...png files from all years…]
│   │   │   │   │   │   └── svgs/
│   │   │   │   │   │   │   └── [...svg files from all years…]
│   │   │   │   └── source/
│   │   │   │   │   └── rule_set_network_viz.py/
│   │   └── interdependency_networks/
│   │   │   │   ├── 1918_interdependency_network.svg
│   │   │   │   ├── 1918_interdependency_network.pdf
│   │   │   │   ├── 1962_interdependency_network.svg
│   │   │   │   ├── 1962_interdependency_network.pdf
│   │   │   │   ├── 2019_interdependency_network.svg
│   │   │   │   ├── 2019_interdependency_network.pdf
│   │   │   │   └── README.md
│
├── manuscript.pdf
│
└── README.md
```