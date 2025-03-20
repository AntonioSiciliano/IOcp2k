using Printf

function pair_correlation_function(type1, type2, r, dr, N_at_range, atoms_types, distances, V, N_at_tot)

    @printf("Julia\n")
    # @printf("Type I %s\n", type1)
    # @printf("Type II %s\n", type2)
    # @printf("The r value is %f\n", r)
    # @printf("The dr value is %f\n", dr)
    # @printf("The N at value is \n")
    # println(N_at_range)
    # @printf("The atoms types are %s\n", atoms_types)
    # @printf("The distances are \n")
    # println(distances)
    # @printf("The volume %f\n", V)
    g_r = 0.0  
    for atom1 in N_at_range
        for atom2 in N_at_range
            if atom1 != atom2
                # println(atom1, atom2)
                if (atoms_types[atom1] == type1  &&  atoms_types[atom2] == type2) || (atoms_types[atom2] == type1  &&  atoms_types[atom1] == type2)
                    d = distances[atom1,atom2]

                    if (r - dr) < d && d < (r + dr)
                        println(atom1)
                        g_r += dr * V/(4 * pi * dr^2 * N_at_tot)
                    end
                end
            end
        end
    end

    return g_r
end