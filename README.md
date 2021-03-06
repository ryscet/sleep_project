Validation of Neuroon startup
-----------------------------
One of most successful Polish startup companies, Neuroon, made a claim they can automatically classify sleep stages and increase sleep quality using a small wearable mask.


We compared the sleep stage classification produced by Neuroon with a clinical standard polysomnography recording.

We have divided the Neuroon validation into three stages:

1. [First](https://github.com/ryscet/sleep_project/blob/master/Time_synchronization.ipynb) we assessed whether the EEG signal collected with Neuroon is similar at any point in time to the polysomnography EEG signal. 


<img style="float: left;" src="https://github.com/ryscet/sleep_project/blob/master/figures/cross.png">


2. [Second](https://github.com/ryscet/sleep_project/blob/master/Hipnogram_comparison.ipynb) we assessed accuracy with which Neuroon predicted the sleep stage. 


<img  style="float: left;" src="https://github.com/ryscet/sleep_project/blob/master/figures/hipno_cm.png">



3. [Third](https://github.com/ryscet/sleep_project/blob/master/Spectral%20analysis.ipynb) we assessed whether there is enough information in the EEG signal to discriminate on that basis between the sleep stages.



<img  style="float: left;" src="https://github.com/ryscet/sleep_project/blob/master/figures/spectral2.png">

