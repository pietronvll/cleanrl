# POWR + SAC Implementation Notes

1. Adde reg, num_rfs and rfs
2. Adjusted imports
3. imported OrthogonalRandomFeatures (with no modifications)
4. Modified SoftQNetwork init (see diff in github)
5. get_action_value content of agent in powr ppo inside forward
6. Added fit_world_model and compute covariance in SoftQNetwork and covariance function 
7. Added seed in soft q network initialization and creation
8. Changed the q_loss computations and updates with the world model fitting 
9. Added q_targets copy during creation and training
10. I m not sure about compute covariance. Added actor param in fit and compute covariances. Modified line 305, TO CHECK action_mean and action_logstd