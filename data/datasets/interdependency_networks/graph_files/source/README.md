The interdependency networks are saved in the .gexf format, and are aggregated at the level of the Laws. 

There are two ways to count the citations, `single_count` and `multi_count`. This governs how the citations are counted when a rule in one Law cites multiple rules in another Law. 

To illustrate how they work differently—if Law 24.3.2 cites Laws 25.2, 25.3 and 25.4, then the `single_count` setting will increment the weight of the directed edge from Law 24 to Law 25 just once, while the `multi_count` setting will increment the weight of the directed edge from Law 24 to Law 25 three times for the three citation instances. 

All analysis on the interdependency networks in the manuscript have been done with the interdependency networks generated under the `multi_count` setting. 