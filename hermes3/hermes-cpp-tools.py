if (sol_recycling) {

    if(mesh->lastX()){
      for(int iy=0; iy < mesh->LocalNy ; iy++){
        for(int iz=0; iz < mesh->LocalNz; iz++){
          Tn(mesh->xend, iy, iz) = 1.0;
          Nn(mesh->xend, iy, iz) = 1.0;
          // std::cout << std::string("Iterating:") << iy << "\n";
        }
      }
    }

    for(int ix=0; ix < mesh->LocalNx ; ix++){
      for(int iy=0; iy < mesh->LocalNy ; iy++){
          for(int iz=0; iz < mesh->LocalNz; iz++){

            // output << "("" << ix << "Y:" << iy << "Z:" << iz << "T:" << Tn(ix, iy, iz) << "  ";
            std::string string_count = std::string("(") + std::to_string(ix) + std::string(",") + std::to_string(iy)+ std::string(",") + std::to_string(iz) + std::string(")");
            output << string_count + std::string(": ") + std::to_string(Tn(ix,iy,iz)) + std::string("; ");
          }
      }
    output << "\n";
    }

  }
  
  output << std::string("\n******************************************\n");
  output << s1->first << s2->first << std::string(": ") << collision_rates[s1->first][s2->first];
  output << std::string("\n******************************************");


for(int ix=0; ix < mesh->LocalNx ; ix++){
      for(int iy=0; iy < mesh->LocalNy ; iy++){
          for(int iz=0; iz < mesh->LocalNz; iz++){

            BoutReal gx = mesh->getGlobalXIndex(ix);
            BoutReal gy = mesh->getGlobalYIndex(iy);
            BoutReal gz = mesh->getGlobalZIndex(iz);

            if (gx == 3) {

              output << nu_12(ix,iy,iz) << "\n";
            }
            
            // output <<"MYPE: " << mype << ", (" << gx << ", " << gy << ", " << gz << "), nu_12 = " << nu_12(ix,iy,iz);
            // if ((gy > 39.5) and (gy <40.5)) {

            //   std::string string_count = std::string("(") + std::to_string(gx) + std::string(")");
            //   output << string_count + std::string(": ") + std::to_string(nu_12(ix,iy,iz)) + std::string("; ");

            // }
            // output << "("" << ix << "Y:" << iy << "Z:" << iz << "T:" << Tn(ix, iy, iz) << "  ";
            
          }
      }
          
          # SETS WIDTH (precision?)
if (gx == 3) {
              output << std::setw(10) << nu_12(ix,iy,iz) << "\t" << nu(ix, iy, iz) << "\n";
            }