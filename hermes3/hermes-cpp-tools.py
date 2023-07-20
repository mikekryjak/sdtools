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