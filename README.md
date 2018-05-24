# moritanaka
The Mori-Tanaka model is a popular method for homogenizing short-fiber composites.
It is considered a Mean Field theory as it uses Mean Field averaging of the rigorous elasticity solution developed by Eshelby to predict stiffness at realistic volume fractions (Eshelby's method is only valid for very dilute concentrations).

In this calculation, the Mori-Tanaka model itself is only used to calculate uni-directional properties, while the Orientation Averaging procedure described by Advani and Tucker is used to find the stiffness.

The Orientation Averaging uses an orientation tensor to calculate the oriented stiffness tensor. 
In practice, the second order orientation tensor is most commonly used, but it is not sufficient to determine the stiffness (a fourth-order tensor property).
The user may either choose a method to approximate the fourth-order orientation tensor (referred to in the literature as closure models) or enter the fourth-order orientation tensor directly, if it is available.
