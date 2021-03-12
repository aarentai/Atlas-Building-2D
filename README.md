## Algorithm trick
1. In energy calculation, only use the binary mask, rather than weighted map, which will change the alpha field applied to the tensor field previously and result in geodesic going wrong.
2. Both metric matching and mean calculating should be implemented on the inverse of the original DTI tensor field, since the geodesics are running on the inverse of the tensor field.
3. When accumulating of diffeomorphism, always remember the order of accumulation of phi and its inverse is different.
    * `phi_acc = compose_function(phi_acc, phi)`
    * `psi_inv_acc = compose_function(phi_inv, psi_inv_acc)`
## Data processing and plotting trick
4. As for the `x` and `y` ambiguity in indexing and ploting, calling the first two dimension in `[145, 174, 2, 2]` in the order of x, y is the best choice! 
    * When indexing the array, `x` indexes row and `y` indexes column, the way I typically do and the way how matplotlib plot the 2d image. 
    * When plotting the tensors, matplotlib would rotate the array counterclockwise by 90 degrees. So the vertical axis is `y` and horizontal axis is `x`, which is also consistent with my knowledge in drawing the Cartesian coordinate system. Fortunately, kirs' code has already done in this way, like the ellipse(x, y). 
5. The tensor argments in all functions, are of size like `[145, 174, 2, 2]` should always make the last two dimension index metric matrix, to comply the how pytorch works.
6. When read the file like cubic in data, remember to permute the dimension like (2,1,0), which is very trick due to the design of itk. Note that you have to be very careful of ordering conventions.  SimpleITK/sitk (which is nice for reading/writing images etc) will store things in an x,y,z, but when changing to/from numpy, sitk will change the ordering so that numpy is z,y,x.   Blake chose to leave it this way so that his numpy and pytorch orderings are z,y,x.

In the code I provide here, I instead created a method GetNPArrayFromSITK and GetSITKImageFromNP that keep the x,y,z ordering when going back and forth.  Thus, in my code below, I stay in x,y,z ordering everywhere.
7. geo.geodesicpath's input should be original DTI tensor fields, instead of the inverse