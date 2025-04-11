
function get_time_correlation_vector(V, N)
    """
    GET THE TIME CORRELATION USING THE SCALAR PRODUCT AMONG VECTORS
    ===============================================================
    """
    
    println("\n ==>WELCOME TO JULIA...")
    # println("WELCOME TO JULIA...")
    # println("Hello from Julia! x = ", typeof(V))
    println(" ==>Computing the windowed average of a trajectory with size ", N)

    # The correlations and the corresponding errors
    correlations = zeros(Float64, N)
    error_correlations = zeros(Float64, N)

    # A dummy variable
    value = 0.
    
    for m in 1 : N
        # println(" ", m)
        all_products = zeros(Float64, N - m)
        for i in 1 : (N - m)
            # println("  ", i)
            # The scalar product
            value = V[i, :]' * V[i + m, :] 
            correlations[m] += value /(N - m) 
            all_products[i] = value
        end

        error_correlations[m] = sqrt(sum((all_products .- correlations[m]).^2)) / (N - m)
        # println(J[m,:]) np.sqrt(np.sum(all_orducts - correlations[m]))/(N - m)
    end

    # println(" c", correlations)
    println("\n ==>BYE JULIA...")
    return correlations, error_correlations
end