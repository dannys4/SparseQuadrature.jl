using Random
using SparseQuadrature: SortedList

@testset "Int payload" begin
    list_unsorted = [3,2,5,4,1,7,-1,0]
    lt = <
    list = SortedList{Int}(lt)
    for el in list_unsorted
        push!(list, el)
    end
    list_sorted = collect(list)
    list_sorted_ref = sort(list_unsorted;lt)
    @test list_sorted_ref == list_sorted
    list_sorted = []
    for j in eachindex(list_sorted_ref)
        el_j_peek = peek(list)
        el_j = pop!(list)
        @test el_j == el_j_peek
        push!(list_sorted, el_j)
    end
    @test list_sorted_ref == list_sorted
    @test isempty(list)
    @test isnothing(pop!(list))
end

@testset "Tuple payload" begin
    N = 20
    rng = Xoshiro(204820)
    nums = rand(rng, N)
    lt = (x,y)->x[2]>y[2]
    list = SortedList{Tuple{Int,Float64}}(lt)
    for j in eachindex(nums)
        push!(list, (j, nums[j]))
    end
    list_sorted = collect(list)
    list_sorted_ref = sort(collect(zip(eachindex(nums),nums));lt)
    @test list_sorted == list_sorted_ref
    list_sorted = []
    for j in eachindex(list_sorted_ref)
        el_j_peek = peek(list)
        el_j = pop!(list)
        @test el_j == el_j_peek
        push!(list_sorted, el_j)
    end
    @test list_sorted_ref == list_sorted
    @test isempty(list)
    @test isnothing(pop!(list))
end