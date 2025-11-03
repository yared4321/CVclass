# CVclass

assignment 1 :

1. Build a system of equations of the form 𝐴𝑥 = 𝑏, as learned in class, for projective
transformation. Attach the formula development to your exercise solution. How do we get the
conversion matrix from the equation system?
2. Write a function that estimates the transformation coefficients from source (src) to destination
(dst), from the equation system in section 1.

Answer:

1. Ax = b when:
   A = [[h00 h01 h02]
       [h10 h11 h12]
       [h20 h21 h22]]
   b = X_hat(without normalize)  = [x_h_raw
                                   y_h_raw
                                   w  ]
   x = source point  = [x
                        y
                        1  ]

   w = h20*x + h21*y+ h22 
   x_dst = x_h_raw/w = h00*x + h01*y + h02*1/h20*x + h21*y+ h22 
   y_dst = y_h_raw/w = h10*x + h11*y+ h12*1 / h20*x + h21*y+ h22

   arrangment 

   x_dst*(h20*x + h21*y+ h22) = h00*x + h01*y + h02*1
   y_dst*(h20*x + h21*y+ h22) = h10*x + h11*y+ h12*1
   
   x_dst*(h20*x + h21*y+ h22) - (h00*x + h01*y + h02*1)= 0
   y_dst*(h20*x + h21*y+ h22) - (h00*x + h01*y + h02*1)= 0

  h20*x_dst*x + h21*x_dst*y + h22*x_dst - h00*x - h01*y - h02*1 = 0
  h20*x_dst*x + h21*x_dst*y + h22*x_dst - h00*x - h01*y - h02*1 = 0
   

   
