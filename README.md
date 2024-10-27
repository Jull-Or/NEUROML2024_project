# Assessment of Video Game Player Proficiency Through the Brain Activity Analysis Using EEG Signals

## Context and Motivation

This research project is dedicated to the study of human brain activity in the context of video games. As a popular digital form of entertainment activity, video gaming has transformed into the rapidly growing electronic sport industry involving single players or teams competing in various tournaments.\
According to the statistics for the current year the global gaming market size is nearly 3 hundred billion dollars ($300 billion). Almost half of the world’s population plays games, and more and more casual gamers strive to make a career as eSports athletes.\
The growing interest in competitive gaming rises to a number of challenges for players on how to progress from the amateur gaming level into the professional one and how one can characterize a professional eSports athlete and his/her skills.\
More specifically, this project is focused on the analysis of the Counter-Strike game, as the most popular in its genre, whose number of active players is almost 2 million at the same time.
Video games have an inherently digital nature and provide easy access to a large amount of structured and unstructured data from various sources, which stimulates scientific interest.\
To date, the vast majority of studies to characterize and assess the players are based on the study of gaming skills and performance metrics based on in-game data, as well as physical skills and physiological patterns based on data from peripheral sensors. Only a few works go deeper and consider the underlying brain activities that contribute to overall player behavior.

## Aim and Objectives

The aim of this project to investigate differences in brain activity between players of different skill levels and to assess the impact of high game proficiency on brain activity patterns.

The objectives run as follows:
1.	Build a dataset consisting of cleaned, filtered, and class-separated EEG signals.
2.	Determine the distinctive patterns of brain activity for casual and professional players. 
3.	Develop a reliable baseline for player skill prediction.

## Methodology

*Data description*

The data used in this project was collected from 17 players with different gaming experience, including 4 professionals, during gaming sessions lasting 30-50 minutes.\
EEG data were recorded using the wireless EEG headset Emotiv Epoc+ from 14 usable saline electrodes according to the 10–20 system (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4) and 2 references on parietal sites (P3 and P4). For some reason, instead of O1, the Pz channel was recorded. \
The data stream from the headset represents automatically pre-filtered bands' power in time for 5 different frequency bands:
*	theta (4-8Hz)
*	alpha (8-12Hz)
*	betaL (low beta, 12-16Hz)
*	betaH (high beta, 16-25Hz)
*	gamma (25-45Hz
  
Detailed technical documentation can be found at: https://www.emotiv.com/products/epoc?srsltid=AfmBOopwvKfkjwR8bie8yF0JP3vozYoYsGhl8i8qXhS6ZkZ9kbPJqtsJ

*Data preprocessing*

Since the data undergoes built-in filtering and is already presented as frequency bands power, additional filtering is omitted.
Preprocessing consisted of:
1)	resampling to a unified sampling rate;
2)	removing outliers;
3)	windowing.

*Feature extraction*

In this work, 2 subgroups of features are extracted - basic time-domain features and autonomic EEG indices.\
Time-domain features were extracted per channel and include Minimum, Maximum, Mean, Std, 1st Quartile, 2nd Quartile (Median), 3rd Quartile, Hjorth parameters (complexity, mobility), Skewness, Kurtosis, Root mean square, Decorrelation time, Peak-to-peak (PTP) amplitude, Katz Fractal Dimension, Area under the curve, Variance, Entropy.\
Autonomic EEG indices tend to measure asymmetry in brain activity in specific frequency bands and express to human preferences and behavior.\
The following are used in this work:\
* Approach-Withdrawal (AW) Index - measures the frontal alpha asymmetry and serves for the evaluation of engagement/disengagement towards the stimuli (in this case - game)
* Effort Index - reflects the activity level of the frontal theta in the prefrontal cortex and serves as the indication of cognitive processing. Higher theta activity associates with higher levels of task difficulty and complexity in the frontal area.
* Valence Index - measures the asymmetrical activation of the frontal hemisphere that correlates to player preferences.
* Choice Index - measures the frontal irregular fluctuations in beta and gamma, associated with the actual stage of decision-making.
All of the above are based on frontal area channels. Similar indices were measured for the temporal cortex.

*Data analysis*

Data analysis involved the following steps: 
1. training on time-domain features -> best model selection
2. adding autonomic features and evaluating the best model from the previous step
3. testing dimensionality reduction and feature selection methods, as well as the upsampling for balancing the class distribution
4. analyzing the model results and feature importances

## Results

It was find out that the best prediction of players' skills was achieved using the Random Forest classifier with the following parameters [ccp=0.001, criterion='gini', max_depth=5, max_features='sqrt'] on mixed features with preliminary application of SelectKBest to select 500 most significant features and subsequent use ADASYN upsampling method to increase the minority/majority class ratio to 0.65. 

According to the feature importance analysis, the most distinctive patterns of brain activity of casual and pro players are in the gamma, beta, and theta frequency bands, as well as in the temporal cortex, the frontal lobe, and the parietal cortex. In general, it can be concluded that i) increased concentration and focus on the sound environment and sound accompaniment in the game (channels T7, P7, T8, P8, rhythm bands Gamma and Beta) reflects a low level of gaming skill, and vice versa, concentration on processing visual stimuli in the game, and coordinated motor actions in response to them are characteristic of pro players. (channels Pz, O2, rhythm Theta).

