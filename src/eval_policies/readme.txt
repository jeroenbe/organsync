This is an adapted reimplementation of LivSim
* We employ the same logic as used in LivSim
* We extend it to allow more general evaluation 
* We remove dependence on geolocations, and leave that for future work
* We provide a Policy interface, easy to implement custom policies for evaluation
* This reimplementation uses Lightning DataModules (see src/data/data_module.py)

