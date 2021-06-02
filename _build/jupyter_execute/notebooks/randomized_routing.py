#!/usr/bin/env python
# coding: utf-8

# # **Randomized Routing** #
# ## Introduction 
#   Traditionally, algorithms have been designed to improve/bound the worst case performance on a problem. Such algorithms draw deep insights into the problem and carefully design and use datastructures. On the other hand, there are simpler algorithms which work very well on an average input while performing much worse on smaller number of bad inputs. QuickSort is a classic example of such an algorithm for sorting. While MergeSort has worst case complexity of $O(nlogn)$, QuickSort performs $O(nlogn)$ on random input and can perform upto $O(n^2)$ on *bad* inputs. On average inputs, QuickSort is *quicker* than MergeSort with lesser memory requirement, inplace operations and, hence, better cache utilization ([see link](https://www.geeksforgeeks.org/quicksort-better-mergesort/)). Can the issue of *bad* inputs be solved to avoid worst case performance? As it turns out, the answer to *almost* solving the problem is Randomization. *Bad* inputs for QuickSort turn out to be ones when the elements are already ordered. So what if we randomly shuffle the array before running QuickSort? A simple trick like this would actually resolve the issue of bad inputs. In implementation, [Randomized QuickSort](https://en.wikipedia.org/wiki/Quicksort) randomly partitions the array (by choosing random pivot) and has an expected complexity (over random pivots) of $O(nlogn)$ for **any input**. Though worst case is still $O(n^2)$, note that for this algorithm, there are no *bad* inputs; only bad pivot choices.
# 
#   As a general principle, we can utilize simpler algorithms which work well on an *average / random / uniform* inputs to solve the original problem with no restrictions on inputs. Randomized routing is another classic example. Here, we utilize an algorithm that is already good when the input is uniformly distributed. To make this algorithm work for generic input, all we require is clever randomization that tranforms the problem on any given input to (multiple) instances of the same problem with random input.
# 
# In this lecture, we study the problem of routing messages in distributed computation. In distributed computing, the idea is to distribute computation to different nodes on cluster and consolidate individual outputs to achieve original computation. In the process, messages have to be passed among nodes on the cluster. If the traffic on the cluster is large (i.e. there is congestion), there can be significant delays in message passing and will cause the overall computation to take longer. The task is to minimize the delays caused by congestion. We show how randomization helps in reducing the congestion delays. For simplicity of analysis, we assume Grid network architecture and permutation messaging paradigm of communication. However, the general usage of randomization in routing goes beyond these simplistic assumptions. 

# ## Grid/Hypercube Architecture 
# In very large clusters, every node cannot be connected to every other node. In such a case, nodes are connected to few other nodes and when a message has to be passed from one node to other, it has to determine a path to traverse. We consider a specific arrangement of nodes called grid model. Consider that we have $N=2^n$ nodes in our graph. We identify each of the nodes with binary strings. We would need $log_2(N)=n$ bits to identify N different nodes. In grid model, the two nodes are connected only if they have hamming distance exactly 1; i.e. their binary string ids have only one bit mismatch. So each node hash a out degree equal to number of bits in its representation , i.e. n. The message can be passed in any direction. Hence total degree is 2n. We can see that the total number of directed edges in the graph view of Grid model is nN . In figure 1 a grid model for N=8 is shown.
# 
# ![HyperCube Structure](https://drive.google.com/uc?id=1IrkAr6Go4VseRH3gYB7pYlYlzVx_uB7g)
# 
# 
# 
# 

# ## Routing Problem 
# In this problem, we assume that each link can carry only one message at a time. Hence whenever two or more messages arrive at a link, a queue is formed which is processed squentially, i.e., one at a time. This adds delay to the overall all-to-all communication step. The purpose of any good algorithm is to minimize the delay. This is a load balancing problem in disguise: we want to make sure no link receives overwhelmingly large load. 
# 
# However network routing is extemetly memory constrained. Generally, any network message or packets on have few bits reserved for routing. Essentially, we cannot store any history of the packet traversal. We only have the destination information on each packet. As a result, it is impertative that the routing protocol is memoryless. More specifically, if a packet arrives at any node, just on the basis of its destination, we should be able to determine the next hop (or node) of this packet. This requirement rules out a lot of optimal algorithms. It turns out that there is a simple and clever strtegy, which is memoryless and ensures that no link is travered by the packet twice. 
# 
# ## Bit fixing Algorithm 
# 
# **One-One message passing** : The algorithm can be described in one line: Given the destination of packet *dest* the currect location *curr*, send the packet to the node *next* with all bits same as *curr* except we flip the least significant bit of *curr* which is different from *dest*.  Before delving into the actual problem, its always advisable to look at simpler version of the problem to understand its properties. Lets look at the problem in which we want to pass a message from *start* to *dest* node. *start* and *dest* are bit string identifiers of the nodes respectively. We first present the algorithm and then discuss the properties of this algorithm.
# 

# 
# **Bit Fixing Algorithm (BFA)**
# ```
# Inputs: start , destination
# Output : Path from start to destination
# Algorithm:
#     current = start
#     path = [start]
#     while current != destination:
#         lbit = lowestMismatch(current, destination)
#         next = current
#         next[lbit] = destination[lbit]
#         path.add(next)
#         current = next
#     path.add(destination)
#     return path
#         
# ```
# A sample run of the algorithm is shown in the following figure, where the start=11011 and destination=10101. At each step the first mismatch in bits is underlined.
# 
# <img src="https://drive.google.com/uc?id=1b9LDbzHUqMQ7AK-EwnCrg9n79d8O22dB" width="300" class="center">
# 

# 
# *    **Bit Fixing Algorithm gives shortest path** Path suggested by BFA always has length exactly equal to the number mismatches in the bit representation of *start* and *dest*. Also, we know that each edge joins two nodes that have exactly 1 bit different. Hence length of possible paths have to be greator than or equal to the number of mismatches in *start* and *dest* and BFA outputs the least of them
# 
# *    **Bit fixing Algorithm is memory-less** In order to compute the link on which to send to message, at each step, the algorithm only requires destination address and its own current address (which it knows). So it satisfies the requirement of memory-less we defined above.
# *    **Bit fixing Algorithm has optimal substructure property** If you consider the any sub path (say $node_i$ to $node_j$) of the path BFA outputs for (start, dest) is exactly the path it would output for $start=node_i$ and $dest=node_j$. So in the BFA, there is exactly one path between any two nodes, irrespective of what the start, dest nodes are. 
# *    **Identity of path nodes**  While going from start $(S_1,S_2,\ldots,S_n)$ to dest $(D_1, D_2, \ldots D_n)$, every intermediate node has the bit string representation $(D_1, D_2, \ldots D_i, S_{i+1}, \ldots S_n)$ And with each progressing step the prefix of intermediate node matching with destination only grows longer. This also establishes that there are no cycles in the path.
# *    **Path interaction** The above properties also add restrictions to the interactions of two paths say (s1,d1) and (s2,d2) can have. Specifically, the paths can join each other at at most 1 node and continue on that path till they diverge. They cannot diverge and join each other again. We will use this property in our analysis for randomized routing. This is clearly shown in the following figure
# 
# 
# <img src="https://drive.google.com/uc?id=1wJgBfqjDibm1t09vk2B-DWOqFIy8LVA5" width="800" class="center">
# 
# 

# ## Permutation Routing 
# Now lets look at all-to-all routing. This scheme is also called as permutation routing. For the sequence of start nodes $starts = [1 \ldots N]$, we can specify the destination nodes as a permutation of the start sequence $dests = Perm([1\ldots N])$. We have to pass the message from $starts[i]$ to $dests[i]$ for each i. The following theorem gives an idea of how well a deterministic algorithm can perform. 
# 
# 
# > **Theorem 1:** Any (memory) oblivious deterministic algorithm for permutation routing with N machines and n ($\approx log N$) outward links require $\Omega(\sqrt{\frac{N}{n}})$ steps
# 
# This theorem basically requires us to look for solutions beyond the deterministic realm of algorithms. We will see that the results with the randomized routing will help us break this bound at least probabilistically.
# 
# ### Randomized Routing 
# We will use bitfixing algorithm as a basic routing algorithm. If we were to run bitfixing algorithm on permutation routing problem , how would we expect the algorithm to fare? It is very obvious that an adversarial selection of permutation of starts can cause a lot of delays in routing. However, we would expect the delays to be bounded if the permutation was random. Before moving to the final algorithm, lets evaluate if our intuition is correct. The following theorem, in fact, confirms our conjecture that the basic algorithm for each pair of start and dest nodes already performs very well if the dest is a random permutation.
# 
# > **Theorem 2:** If dests is random permutation of starts, then with probability $(1 - (\frac{1}{2})^{1.5n})$ message from starts[i] reaches the dests[i], for all i, in no more than 4n steps
# 
# ### Solution to general (starts, dests): 
# So from the above theorem, it is clear that our intuition about random destinations is actually correct. We can use this to bound the time of any permutation routing problem. This can be done by adding a random intermediate destination between *starts* and *dests* locations. 
# 
# > starts $\longrightarrow$  dests 
# 
# The above problem can be converted to the following and we use bitfixing algorithm for both the sections of the problem.
# 
# > starts $\longrightarrow$ random $\longrightarrow$ dests
# 
# Using the theorem 2, we can say that with high probability the total time taken by the routing algorithm is bounded by 8n.
# 
# ## General Insight
# It turns out that many algorithms, especially associated with graphs, have this property. When an algorithm is run on a uniformly sampled input (random), the algorithm has good running time with high probability. However, in general, there are many structured inputs, which can very well occur in practice; the performance is terrible. We can make such algorithms work for generic inputs by somehow converting the problem where we only run the algorithm over random inputs and still solve for any given (potentially structured and poorly performing) inputs. The conversion is dependent on the problem. For routing, by enforcing a random permutation of intermediate destinations, we were able to break the original problem with any given source-destination sequence to two problems where either the source or the destination is uniformly shuffled. 
# 
# The classic example is Quicksort , which takes $O(n^2)$ time in the worst case, but when randomized takes $O(n\log{n})$ expected time and the running time depends only on the coin tosses , not on the input. Notice that randomized quick sort always leads to the correct output (Las Vegas Algorithms (https://en.wikipedia.org/wiki/Las_Vegas_algorithm))
# 
# A good lecture note with more such illustration can be found at 
# http://math.mit.edu/~goemans/notes-random.pdf

# ## Code
# In this section, we give a basic delay simulator for routing over hypercube. We encourage the readers to play around with the value of n and different permutations to see how randomized routing successfully breaks the congestion. We also show one way to generate permutation that leads to congestion which is inspired from the constructive proof of Simplified Theorem 1 in Theory Section.
# 
# 
# 
# ## Simulation Code
# each edge has a queue and at each time step only one message can cross over that edge.

# In[1]:


# Simulation code
from collections import deque
import numpy as np
def codearray(ar, n):
    return [codenode(s, n) for s in ar]

def codenode(start,n):
    s = bin(start)[2:]
    s = '0'*(n-len(s))+s
    return s

def linkid(s,d,i,n):
    N = pow(2, n)
    start = int(s, 2)
    dest = int(d, 2)
    val = min(start, dest)
    return val * n + i

def decode(l, n):
    start = l // n
    i = l % n
    return codenode(start,n), i


def bit_fixing_links(start, dest, n):
    N = pow(2,n)
    s = codenode(start, n)
    d = codenode(dest, n)
    links = []
    for i in range(len(s)):
        if s[i] == d[i]:
            continue
        #print(d[:i]+s[i:], "->", d[:i+1]+s[i+1:])
        links.append(linkid(d[:i]+s[i:],d[:i+1]+s[i+1:],i,n))
    return links


def simulate(dests, n): #starts = 0,1,..2^n-1
    N = pow(2, n)
    node_links = {}
    # init node_links node-> set of links
    for i in range(N):
        start = i
        dest = dests[i]
        node_links[start] = deque(bit_fixing_links(start, dest, n)) 
    # init link_queues  link -> waiting messages
    link_queues = {}
    for i in range(N*n):
        link_queues[i] = deque()

    # put first step onto links
    for i in range(N):
        if len(node_links[i]) >0:
            link_queues[node_links[i].popleft()].append(i)
    queues_empty = False
    time = 0
    while not queues_empty:
        queues_empty = True
        nodes = []
        # one message per link
        for i in range(N*n):
            if len(link_queues[i]) > 0:
                node = link_queues[i].popleft()
                nodes.append(node)
                #print("Link", decode(i, n), "message", codenode(node, n), node)
        # process nodes
        for i in nodes:
            if len(node_links[i]) >0:
                link_queues[node_links[i].popleft()].append(i)
        # see if queues are empty
        ct = 0
        for i in range(N*n):
            if len(link_queues[i]) > 0:
                queues_empty = False
                ct = ct + len(link_queues[i])

        time = time + 1
        #print(time, ct)
    return time


# 
# 
# ```
# # This is formatted as code
# ```
# 
# #### Generating Permutation
# In order to generate a permutation that can cause good congestion. We map nodes of the form  0X to X0. where X is of length n/2 bits. Note that all the paths here have to pass through node labelled 0. For the rest of the nodes, we just map them to themselves.

# In[2]:


# Generating cases for routing 

def get_random_perm(n): # random
    N = pow(2, n)
    a = np.arange(N)
    np.random.shuffle(a)
    return a

def get_complementary_perm(n): # complementary X -> not(X)
    N = pow(2, n)
    a = np.arange(N)
    a = [N -1 - i for i in a]
    return a

def get_congested_perm(n): # creates paths that have to pass through node 0 exploiting bit fixing algorithm
    #assert(n%2==0)
    nhalf1 = n//2
    nhalf2 = n - n//2
    N = pow(2, n)
    Nhalf1 = pow(2, nhalf1)
    Nhalf2 = pow(2, nhalf2)
    a = np.arange(N)
    for i in range(Nhalf1):
      mapped_i = i*Nhalf2
      a[i] = mapped_i
      a[mapped_i] = i
    return a


# #### Wrapper for simluation with permutation defined above

# In[3]:


# wrapper for comparison of routing directly vs using randomized intermediate step.
def simulate_delay_congested_paths(n):
    N = pow(2, n)
    print("Number of Nodes:", N)
    c = get_congested_perm(n)
    r = get_random_perm(n)
    print("Running Bitfixing direcly ...")
    dt = simulate(c, n)
    print("steps: ", dt)
    print("Running Bitfixing on starts->random")
    r1 = simulate(r, n)
    print("steps: ", r1)
    a = np.arange(N)
    for i in range(N):
        a[r[i]] = c[i]
    print("Running Bitfixing on random->dest")
    r2 = simulate(a, n)
    print("steps: ", r2)
    print("--------------comparison ---------------------")
    print("No. of steps with Direct:", dt)
    print("No. of steps with Randomized intermediate:",  r1+r2)
    if n < 5:
      print(codearray(np.arange(N), n))
      print(codearray(c, n))


simulate_delay_congested_paths(12)


# ## Play Around with n!
# As expected, the direct routing time increases exponentially with increasing n whereas the randomized routing only increases linearly

# In[4]:


from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

interact_manual(simulate_delay_congested_paths, n=widgets.IntSlider(min=0, max=18, step=1, value=10)); # colab fails >=19 


# ## Theory
# 
# Proof of Theorem 1
# 
# 
# see theorem 2.2 in 
# [uscd scribe on randomized routing ](https://cseweb.ucsd.edu/~slovett/teaching/SP15-CSE190/randomized_routing.pdf)
# 
# Proof of Theorem 2
# 
# > **Theorem 2:** If dests is random permutation of starts, then with probability $(1 - (\frac{1}{2})^{1.5n})$ message from starts[i] reaches the dests[i], for all i, in no more than 4n steps
# 
# We will use the following notation
# 
# * $Time[message[i]]$ : time taken by message from source i
# * $Path(i)$ : path taken by message from source i
# * $delay(i)$ : added delay due to congestion for message from source i
# * $Intersection(i,j)$ : proposition that path(i) and path(j) share an edge
# * $Intersections(i,j)$ : number of edges shared by path(i) and path(j)
# * $T(e)$ : number of nodes whose path share edge e
# * $length(path)$: number of edges in a path
# 
# 
# We can write the time taken by packet i as following. The 
# $$ Time[message[i]] = length(Path(i)) + delay(i) \leq n + delay(i)$$
# where delay is caused by different messages intersecting the path of this message from starts[i] to dests[i]. As defined in the problem above, the delay occurs only when two or more packets want to use the same link. Hence the paths of i and j must intersect for j to add to the delay for i. One important thing to note is that every packet can at most add a delay of 1 unit to the other packet. This is because a packet j can only \textit{join} the path of i at one single point. This follows from the optimal substructure property of BFA . Also, while they are sharing the path, a delay is added only at the start of the shared path.  So we can write the intersection as an indicator.
# $$
#     delay_i \leq \sum_j \mathbb{I}(Intersection(i,j))
# $$
# We can simplify the RHS of the above inequality by noting the following.
# $$
#     \sum_j \mathbb{I}(Intersection(i,j)) \leq \sum_{j} Intersections(i,j)
# $$
# Where we replace the indicator with number of intersections. Also, we can write the RHS of the above inequality in terms of sum over edges as 
# 
# $$
#     \sum_j \mathbb{I}(Intersection(i,j)) \leq \sum_{e \in Path(i)} T(e)
# $$
# T(e) is the total number of nodes sharing the edge e with node i. Note that R.H.S is an overestimate as each Indicator is replaced by the number of edges shared between the two paths.
# Taking Expectation of $delay_i$, we have
# $$
#     E(delay_i) \leq \sum_{e \in Path(i)} E(T(e))
# $$
# As, number of edges is bound by n,
# $$
#     E(delay_i) \leq n E(T(e))
# $$
# in order to find the E(T(e)), consider the following equation counting T(e)
# $$
#     T(e) = \sum_{j \in [1,N]} (\mathbb{I}( e \in Path(j))
# $$
# $
#     \sum_{e \in allEdges} T(e) = \sum_{e \in allEdges} \sum_{j \in [1,N]}  (\mathbb{I}( e \in Path(j)) $
# 
# $ \sum_{e \in allEdges} T(e)  = \sum_{j \in [1,N]} \sum_{e \in allEdges}  (\mathbb{I}( e \in Path(j))$
# 
# $ \sum_{e \in allEdges} T(e)  = \sum_{j \in [1,N]} (length(Path(j))) $
#     
# Taking expectation
# $$ E(\sum_{e \in allEdges} T(e)) = \sum_{j \in [1,N]} E(length(Path(j)))$$
# $$ Nn E(T(e)) = N \frac{n}{2} $$
# $$ E(T(e)) = \frac{1}{2}$$
# 
# Hence expectation of the delay is 
# $$ E(delay_i) \leq \frac{n}{2}$$
# 
# As $delay_i$ is a sum of independent bernoulli variables, we can use chernoff bounds to obtain the tail bound.
# 
# $$
#     P( delay_i \geq (1+\delta) \mu)) \leq e^{-(\frac{\delta^2\mu}{2+\delta})}
# $$
# 
# where $\mu \leq \frac{n}{2} $
# Now we want to choose $\delta$ in a way that rest of the analysis follows, specifically, we want the probability on the right to be small enough to give a good tail bound after we apply the union bound. Lets aim for now to keep the probability on right bounded by $\frac{1}{N^{2.5}}$ 
# $$ e^{-(\frac{\delta^2\mu}{2+\delta})} \leq \frac{1}{N^{2.5}} $$
# 
# $$ -log_2(e)(\frac{\delta^2\mu}{2+\delta}) \leq - 2.5 log_2(N) $$
# 
# $$ log_2(e)(\frac{\delta^2\mu}{2+\delta}) \geq 2.5 n \geq 2.5 * 2 \mu $$
# 
# $$ log_2(e)(\frac{\delta^2}{2+\delta}) \geq 5 $$
#   
# By substituting $\delta = 5$ we can see that the equation above is satisfied.
# Hence we have, 
# $$ P( delay_i \geq 6 \mu)) \leq \frac{1}{N^{2.5}} $$
# 
# $$ P( delay_i \geq 3n)) \leq \frac{1}{N^{2.5}} $$
# 
# Applying union bound over all the nodes, we get
# $$ P( \exists i \quad delay_i \geq 3n)) \leq \frac{1}{N^{1.5}} $$
# 
# Hence the total time according to equation 1, is bounded by
# $$ P( TotalTime \geq 4n)) \leq \frac{1}{N^{1.5}} $$
# Hence with probability $1 - \frac{1}{N^{1.5}} ( = 1 - (\frac{1}{2})^{1.5n})$, the total time taken is no more than 4n \\
# That completes our proof.
