using Base: collect, pop!, push!, peek, isempty

mutable struct SortedListNode{P}
    payload::Union{Nothing,P}
    next::Union{Nothing,SortedListNode{P}}
end

mutable struct SortedList{P,C}
    comp::C
    head::SortedListNode{P}
    function SortedList{_P}(comp::_C) where {_P,_C}
        tail = SortedListNode{_P}(nothing, nothing)
        head = SortedListNode{_P}(nothing, tail)
        new{_P,_C}(comp, head)
    end
end

function Base.push!(list::SortedList{T}, val::T) where {T}
    prev = list.head
    node = prev.next
    while !isnothing(node.payload)
        list.comp(val, node.payload) && break
        prev = node
        node = prev.next
    end
    val == node.payload && return false
    new_node = SortedListNode(val, node)
    prev.next = new_node
    true
end

function Base.pop!(list::SortedList)
    ret_node = list.head.next
    isnothing(ret_node.payload) && return nothing
    list.head.next = ret_node.next
    ret_node.payload
end

function Base.peek(list::SortedList)
    ret_node = list.head.next
    isnothing(ret_node.payload) && return nothing
    ret_node.payload
end

function Base.isempty(list::SortedList)
    isnothing(peek(list))
end

function Base.collect(list::SortedList{T}) where {T}
    ret = T[]
    node = list.head.next
    while !isnothing(node.payload)
        push!(ret, node.payload)
        node = node.next
    end
    ret
end