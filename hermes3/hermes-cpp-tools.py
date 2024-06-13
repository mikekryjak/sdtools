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
  
output<<std::string("\n\n****************************************************\n");
output << std::string("Collisions: ") << species1.name() << species2.name();
output<<std::string("\n****************************************************\n\n");

output<<std::string("\n------------------------\n");
output << std::string("Definitely doing this");
output<<std::string("\n------------------------\n");
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

###################### Options objects
####### Iterate through children of options:

const auto& colls = species["collision_frequencies"].getChildren();

    for (const auto& coll : colls) {
      output << coll.second.name() << std::endl;   # endline
    }

####### Print all options:

output << toString(state["species"]);

###################### TYPES

"""
peter.hill
  2 hours ago
output.print("{}", options); should print an Options to screen. What went wrong when you used toString()?


Mike Kryjak
  2 hours ago
Oh man, 
@peter.hill
 thank you for this - I didn't know there was a whole output class. It looks super useful. I've just been blindly doing output << str without understanding what's going on.
On the toString(), I think I got confused with a BoutData method that prints the tree structure of the entire BOUT.inp options. It looks from the code that toString() actually converts a singular options entry to a string?


peter.hill
  1 hour ago
output << options, output << options.toString(), and output.print("{}", options) all do (essentially) exactly the same thing :slightly_smiling_face:


peter.hill
  1 hour ago
they all convert the whole Options object, including all sections and subsections, to a string and print it


Mike Kryjak
  1 hour ago
I tried this line: output << state["species"].toString();
And I get the below:
In file included from /ssd_scratch/hermes-3/src/hydrogen_charge_exchange.cxx:1:
/ssd_scratch/hermes-3/src/../include/hydrogen_charge_exchange.hxx: In member function 'void HydrogenChargeExchangeIsotope<Isotope1, Isotope2, Kind>::transform(Options&)':
/ssd_scratch/hermes-3/src/../include/hydrogen_charge_exchange.hxx:163:32: error: 'class Options' has no member named 'toString'
  163 |     output << state["species"].toString();
      |                                ^~~~~~~~
make[2]: *** [CMakeFiles/hermes-3-lib.dir/build.make:650: CMakeFiles/hermes-3-lib.dir/src/hydrogen_charge_exchange.cxx.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:140: CMakeFiles/hermes-3-lib.dir/all] Error 2
make: *** [Makefile:146: all] Error 2


Mike Kryjak
  1 hour ago
But using output.print("{}", options) results in a correct print


peter.hill
  1 hour ago
ah, sorry, it's not a method but a free function, so output << toString(state["species"]) would work


Mike Kryjak
  1 hour ago
Oh I see! I've heard of it as a free function but thought it was also a method on its own.
Thanks for the help on this :smile:

"""