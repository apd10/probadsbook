#!/usr/bin/env python
# coding: utf-8

# # **Ideas behind many probabilistic data structures**
# 

#   In this chapter, we would like to discover the fundamental ideas behind a lot of probabilistic datastructures such as bloom filters, count sketches and so on. This chapter will use the problem of set membership as a running problem and we shall try to see where the ideas stem from. In the subsequent chapters we will go in detail on individual topics motivated here. 
# 
#   We want the readers to carefully understand the importance of following ideas in probabilistic data structures
# *   *Sacrificing Correctness in an controlled way can lead to phenomenal improvements in performance giving reasonable solutions to problems that cannot have practical deterministic solutions.*
# *   Apart from this, we will also motivate why hashing comes in the picture and why it warrants a detailed discussion
# *   Median of Means Trick : How simple insights from probability theory are utilized as a key step to making these algorithms really sucessful.
# 
# So lets begin by taking a running example
# 
# 

# ## Problem Statement
# 
#   The general problem we want to solve is of keeping some statistic for each of potentially large objects. The objects can be something like student class object with many string attributes like name, city, contact, educational institutions, performance scores etc. It can also be large webpages on internet or simply the url address of a webpage. Basically, the object is any data required for identifying a required entity. statistic, on the other hand, is a value you want to store against each object. For example, in case or url object, you could want to store the number of hits each url got. Or in case you want to maintain a set of malicious urls, the statistic can be as simple as maintaining a boolean value. 
# 
#   As a running example, we will try and look at a set of malicious urls. The idea is to have a datastructure storing these urls. We can add new urls to this set. At anytime, we can query this datastructure to check if a url is malicious. Ofcourse, the number of queries can be very large, so we have to keep query time into consideration. Also, as webbrowser needs to be light, we have to keep memory used into check as well. To set realistic limits, lets say
# 
# 
# > * Design a datastructure to answer a query : *is a url malicious?*. 
# > * You have $10^6$ queries of average length 50 chars each.
# > * QueryTime : O(1)
# > * Memory : 4MB
# 
# 
#   Lets try to understand the requirements of the problem. 
#   * If we are to store all the urls in any manner, it would take  $10^6 \times 50 \times 8 $ bits i.e. 50 MB memory. So storing the entire data is out of question. 
#   * Nothing is mentioned on preprocessing time. So maybe we can do a lot of intelligent preprocessing (even though this is true for this particular problem statement. This is not generally true. These datastructures generally also have an *update* interface which means datastructure is built out of series of updates and updateTime should also be O(1)). 
#   * The query time has to be O(1) so even the best datastructure to store a set , i.e. balanced binary tree  O(log(n)) is too time consuming. 
# 
# So just looking at this problem statement from the view of traditional algorithms makes it clear that this problem **just cannot be solved**.
# 
# Lets try building a **reasonable** solution!

# ## Building a Solution
# 
# ### Notations
# * $n$ : number of urls
# * $D$ : size of each url
# * $s$ : size of statistic (1 bit for boolean, 32bits for int etc)
# * $N$ : range of the object if mapped to integers
# 
# Each object can be viewed as a binary string which, in turn, can be interpreted as an integer. These can get very large depending on an object. For example the string of len 50 has 50*8 bits. i.e. the integer corresponding to it will be around $2^{400} \sim 10^{120}$. So in our problem you can imagine n objects being drawn from a huge range $[0,N]$
# 
# You can also see that N and D are actually related. D ~ log(N) as D is exactly the number of bits required to store the object which is integer of order of N
# 
# ### How do existing datastructures fare
# 
# Lets just have a look at different datastrucutres, their memory usage and query times.
# * Arrays : 
#   * query time O(1)
#   * Memory O(Ns)
# 
# * Linked List:
#   * query time  O(n)
#   * Memory O(n(D+s))
# 
# * Balanced Binary Tree:
#   * query time O(log(n))
#   * Memory O(n(D+s))
# 
# * Aim:
#   * query time O(1)
#   * Memory O(ns)
# 
# It seems that Arrays, with their indexing magic, are able to give us O(1) query time. However the amount of Memory they need is just ridiculous. On the surface, it seems like we are not storing the `data'. However, it is worse than storing data which is O(logN). It is like having a one-hot encoding of an integer instead of binary. None the less, it begs a querstion : if we are interested in only n cells of this array, is there a way to only keep n or O(n) in contiguous locations and make this work? Can we achieve O(1) query time?
# 
# 
# 
# 
# 
# 
# 
# 
# 

# ### **Idea 1 : Address space reduction : hash maps**
# 
#  We want Array but only O(n) sized. Clearly the naive indexing is at fault. If we use the binary representation of the object, it is bound to explode in memory. We need a clever indexing. Specifically we want a function that maps the object to a smaller range of O(n), say m. So lets consider a function H
#  > $H: [0,N] \rightarrow [0,m] $
# 
#  In literature, these are called as hashing functions. 
# 
# Lets explore this idea 
# 
# * If we are able to match each of the url in the set to a **unique** integer in [0,m], then we can get correct result without having to store the data itself. As we will see later, for a fixed set, we can come up with such a mapping. It is called *perfect hashing*. However, it is impossible to have such a function without knowing the data before hand. In fact, pigeon hole principal clearly dictates that for any fixed function, there exists data which will not lead to unique mappings.
# 
# * So there can be *collisions* of objects at a single mapping into the array. How do we resolve these collisions? There are many strategies used in the hash map datastructures to resolve these collisions like chaining / probing. However, all these datastructure force you to store the data for actual comparison before. Lets say for now we use chaining. It has a linked list stored at each location in the array of size m. So with this idea, we are still stuck with storing entire data.
# 
# A hash map with chaining looks like:
# 
# ![taken from https://i.stack.imgur.com/BrWiZ.png ](https://i.stack.imgur.com/BrWiZ.png)
# * Query time for a particular url u depends on the length of the linked list at the location H(q). We would want that this list is as small as possible. We also note that the length of this list depends on the hash function we choose.
# 
# * How do we choose a hash function?
#   * We can see that any fixed hash function can have an adverserial data which can cause all the data to land at a single point. So the worst case queryTime always remains O(n)
#   * How can we break the adversary? We randomly select function H from a set of hash functions (also called Family of Hash Functions) independent of the data. This is one of the classic ideas used to deal with adversarial data. This is also the place where randomness crawls into the algorithm.
#   * What kind of family (HF) would do? We want the collisions to be less. Mathematically, we can say that we want collision probability to be low. In the best case, the hash function is completely random. i.e. Hash family is set of all mappings from $[0,N] \rightarrow [0, n]$. So it independently maps each of the object. Then collision probability is 
#   $$P_{H \in HF }(collision(o1, o2)) = 1/m $$
# 
#   * Storing the hash function should not be a significant overhead. If we are to store a hash function from a set of all mappings, it would require us to store the entire mapping for each element in $[0,N]$ which is O(N). So this family won't do. There has been research into hash families that still achieve something reasonable, though weaker, in terms of probabilites and require O(1) space to store. We defer detailed discussion on these families for later. For now, we just want you to know that families called universal family exist where
#     * Memory is O(1)
#     * $$P_{H \in HF }(collision(o1, o2)) \leq 1/m $$
#   
#       Note that the completely random hash family is much stronger than this family. For example, in random family $P_{H \in HF }(collision(o1, o2, o3, .. ok)) \leq 1/m^{k-1} $. However universal family does not provide such guarantee.
# 
# 
# ![Choosing H from family of hash functions](https://drive.google.com/uc?export=view&id=15wq7An6Pg5QWhEkPTAKITNksqdVtCHg_)
# 
# 
# * Query Time warrants more discussion. We can see that the randomization trick does not solve the problem of worst case time complexity. Is that the end of it? We believe, we have setup the hash functions so that the query time *practically* is very good. Lets try to capture this intuition. The query time depends on the length of the linked lists. i.e. on the number of elements mapped to the same location where query q lands
# The estimator of the length can be written as 
# $$Length(q) = \Sigma_{i \in n} I(H(q) == H(o_i))$$
# 
# where I is the indicator function.
# If we look at the expectation and use universal hash family
# $$E(Length(q)) = E(\Sigma_{i \in n} I(H(q) == H(o_i)))$$
# 
# $$E(Length(q)) = \Sigma_{i \in n} P(Collision(q, o_i))$$
# 
# $$E(Length(q)) = \frac{n}{m}$$
# 
# we choose m = O(n) so we can see that the 
# $$E(Length(q)) = O(1) \implies E(queryTime) = O(1)$$
# 
# In expectation we see that query time is actually O(1). So it validates that we indeed have achieved something. Also, it points out that more tools are needed to analyse these algorithms now that we have introduced randomness. We should also note that Expectation may not be the correct metric. The actual thing we want to have is that the probability of length of linked list being large should be bounded by a very small quantity. This brings us into discussion of tail bounds on estimator values. A good resource on tail bounds is [tail bounds by Luay Nakhleh](https://www.cs.rice.edu/~as143/COMP480_580_Spring20/slides/TailBounds.pdf)
# 
# 
# 
# What have we achieved
# * Memory O(n(D+s))
# * Querytime E(queryTime) = O(1). For now lets assume this is good enough.
# 
# So we have made significant improvement in querytime. 

# ## **Idea 2: Sacrificing Accuracy to save memory.**
# 
# We saw that just storing the data will lead us to use 50MB of memory. However, because of the unavoidable collisions, we have to keep the data if we want to always provide the correct answer. But, we have also made sure that our choice of hash function reasonably distributes the elements across the array. If we could ensure perfect hashing, we would not need to store the elements and the memory would come down to O(ms). If we remove storing the data, we would not be able to distinguish between the elements mapped to single location. There will be error but is it significant? Can we accept that error? This is one of the key ideas behind randomized algorithms. As we will see, this out of the box thinking lead us to achieve something phenomenal
# 
# ![](https://drive.google.com/uc?export=view&id=1z-_Y_0tPVBQKnFn2WjoZyNK5Q15o9Y2D)
# 
# So in this problem, we scrap all the data stored in linked list. Instead of it, we only store a single bit to denote if any element from the set mapped to this location. Assuming completely random functions, lets try to see how much error can this really lead do. 
# 
# #### Error due to collisions
# 
# Firstly, we would like to note that the datastructure can only report false positives (An element not inserted is reported as a member). It would never report false negatives. (An element inserted is always reported as member. So let us see what is the probability that a query leads to false positive. 
# 
# Let q be a non-member of the set. 
# 
# $$P(Error) = P(H(q) == 1 |  \textrm{q is not inserted}) $$
# 
# $$ = P(\textrm{at least 1 insertion lands at } H(q))$$
# 
# $$ = 1  - P(\textrm{no insertion at }H(q))$$
# 
# $$= 1 - \Pi_i P(H(e_i) != H(q))$$
# 
# $$1 - \Pi_i (1 - P(H(e_i) == H(q)))$$
# 
# $$1 - \Pi_i (1 - \frac{1}{m})$$
# 
# $$1 - (1 - \frac{1}{m})^n$$
# 
# $$\leq 1 - e^{-\frac{n}{m}} \quad  \textrm{ using }( 1 - 1/x \leq e^{-1/x})$$
# 
# We see how false positive rates for some values of m in the table below.
# 
# | Method         | Memory  | Error |
# |----------------|---------|:-----:|
# | hashmap        | 50MB    | 0     |
# | bitarray m=5n  | 0.625MB |  0.18 |
# | bitarray m=10n |  1.25MB |  0.09 |
# | bitarray m=20n |   2.5MB |  0.05 |
# 
# 
# 
# It seems like we have some reasonable solution here for the original problem we started with. If we want to improve the result further, one clear way is to increase the value of m , i.e. increase the memory used. There are a couple of points to note here
# 
# * The error is going down sub-linearly with increasing memory. 
# * If under the same setup we had to store some integer / float statistic, the memory required (32 / 64 bits instead of 1 bit) would not seem as an impressive improvement. Ofcourse, this does not discredit the algorithm's general improvement in memory usage. 
# 
# So, there is a need for a better way of reducing the error than just increasing memory.
# 
# 
# 
# 
# 
# 
# 

# ## Idea 3 : **Repeat and Combine** to reduce error
# 
# ---
# 
# 
# 
# Before we move further, lets look at a very basic tool to improve performance in probabilistic processes. Consider an urn with 80 Red balls and 20 white balls (each numbered 1 to 100). Your goal is to report a number of a ball which is red; but you cannot look at the urn or the balls. You have a verifier who can verify the colors corresponding to the number / numbers you ask. However you can only ask the verifier once. What is a good strategy? You can sample a number randomly from 1 to 100 and ask the verifier for the color. You know you can end up with the wrong ball with probability 0.2 . A good way to improve your chances of getting a red ball is to sample more numbers and ask verifier to verify them. If you sample 2 balls, the probability of not having a red ball is reduced to 0.2^2 = 0.04. Similarly, if you sample n times, the probability failure is 0.2^n. As you can see by increasing the processing cost linearly (for sampling and verifying) you can **exponentially** reduce the probability of error. Note what we did was
# * We *repeated* the process of sampling multiple times. It gave us n answers each of which had some probability of failure.
# * We *combined* the answers (in this case it was simple to just see and pick red ball) to give the best answer.
# 
# This repeat and combine technique is prevalent in a lot of algorithms in sketching literature and is used to reduce the error bound exponentially.
# 
# Lets go back to what have we achieved with a single bit map. We create a bit map which a randomly drawn hash function. It gives us the probability of error on any q to be bound by some error $e$. You can see that the begin correct/wrong is a random variable that comes out of the random process of building a bit map seeded by a random draw of hash function. So lets try to apply *repeat and combine* technique here. We choose another independently drawn hash function and that will give us another bit map which also has same error bounds. Now can we combine the results for a query q. For query q we get two values of is_set from two arrays. We know that there can only be false positives. So if any one of them is false, we know that q was never inserted. So we answer (is_set = is_set_1 AND is_set_2). 
# $$P(Error) = P(A1[h_1(q)] \wedge A2[h_2(q)] | q \not \in S)$$
# 
# $$P(Error) = P(A1[h_1(q)] | q \not \in S) P(A2[h_2(q)] | q \not \in S) = P(A1[h_1(q)] | q \not \in S)^2 $$
# 
# $$P(Error) = e^2 $$
# 
# For r repetitions, the memory required here would be O(rms). for rm=20n we compare for different values of r and n in the table below:
# 
# | Method                   | Memory  | Error |
# |--------------------------|---------|:-----:|
# | hashmap                  |   50MB  | 0     |
# | bitarray m=n r=20        |   2.5MB |  0.0001 |
# | bitarray m=2n r=10       |   2.5MB |  0.00008 |
# | bitarray m=4n r=5        |   2.5MB |  0.0005 |
# | bitarray m=5n r=4        |   2.5MB |  0.001 |
# | bitarray m=10n r=2       |   2.5MB |  0.009 |
# | bitarray m=20n r=1       |   2.5MB |  0.05 |
# 
# As we can see, we achieve the best results at m=2n and r=10 here. We can see two opposing errors here. We can see that the error decreases as both m and r increase. However due to relation mr = constant. increasing one causes other to decrease. This makes an optimum solution for m and r possible. 
# The final datastructure : r repetitions of m sized bit arrays which have independently drawn hash functions is very close to how a bloom filter is implemented with some modifications. 
# 
# 
# 
# 
# 
# 
# 
# 

# ## Bloom filters, Count Sketches, Count Min sketches
# 
# All the  three datastructures: bloom filters, count sketches and count min sketches, deploy same tools which are discussed below and only differ in the type of statistic stored against each object. The type of statistic also affects the way to combine different repititions. (Note, in bloom filters instead of having repititions as independent memory arrays, we use a single array and use r hash functinos to hash into it. This gives slightly better bounds on error)
# 
# We list the type of statistic and method to combine in the table below
# 
# | DataStructure         | Type(statistic)  | Combination |
# |----------------|---------|:-----:|
# | Bloom filter        |  boolean   |  AND     |
# | count min sketch  | +ve integers/float |  MIN |
# | count sketch |  integers/float |  MEDIAN |
# 
# 

# ## Slides
# The slides on which this is based are here [slides](https://drive.google.com/file/d/1rtfhEXhxXZwkJV0XkRUD2Msd3_GMhH7w/view?usp=sharing)
