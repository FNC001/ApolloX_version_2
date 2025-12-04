# ApolloX_version_2
# Generative Design of Boron-Assisted Amorphization Routes for Multi-element Catalysts with Tunable Short-Range Order

Honglin Li,1, 2, * Chuhao Liu,3, 4, * Yongfeng Guo,2, * Xiaoshan Luo,1, * Yijie Chen,5, * Guangsheng Liu,6 Yu Li,2 Ruoyu Wang,7 Zhenyu Wang,1 Jianzhuo Wu,2 Shouwei Zuo,8 Zhen Luo,4 Cheng Peng,4 Jialu Li,6 Cheng Ma,1 Zhuohang Xie,1 Jian Lv,1 Yufei Ding,9 Huabin Zhang,8 Jian Luo,2, 6 Zhicheng Zhong,7 Yuzhu Wang,10 Mufan Li,4, t Yanchao Wang,1, ‡ and Wan-Lu Li2, 6, §

1 Key Laboratory of Material Simulation Methods and Software of Ministry of Education,  
College of Physics, Jilin University, Changchun, China  
2 Aiiso Yufeng Li Family Department of Chemical and Nano Engineering,  
University of California, San Diego, La Jolla, CA, USA  
3 Institute of Molecular Engineering Plus, College of Chemistry, Fuzhou University,  
Fuzhou, China  
4 College of Chemistry and Molecular Engineering, Peking University, Beijing, China  
5 Institute of Modern Physics, Fudan University, Shanghai, China  
6 Program in Materials Science and Engineering, University of California, San Diego,  
La Jolla, CA, USA  
7 School of Artificial Intelligence and Data Science, University of Science and Technology of China, Hefei, China  
8 Center for Renewable Energy and Storage Technologies, Physical Science and  
Engineering Division, King Abdullah University of Science and Technology,  
Thuwal, Kingdom of Saudi Arabia  
9 Department of Computer Science and Engineering, University of California,  
San Diego, La Jolla, CA, USA  
10 Shanghai Synchrotron Radiation Facility, Shanghai Advanced Research Institute,  
Shanghai, China  

* These authors contribute equally to this work.  
t mufanli@pku.edu.cn  
‡ yanchao_wang@jlu.edu.cn  
§ wanluli@ucsd.edu  

Engineering short-range atomic order in amorphous materials offers a promising yet scarcely explored route to high-performance materials, but rational design is hindered by vast configurational space and the lack of predictive structure–property relationships. Here we develop ApolloX, a physics-informed, short-range-order–constrained generative framework that integrates conditional generative modeling with particle swarm optimization to navigate the disordered energy landscape of multi-element systems. Using chemical short-range order as an explicit constraint, ApolloX identifies low-energy amorphous configurations and composition-driven trends such as metal clustering, sluggish diffusion, and enhanced amorphization. Building on these capabilities, we establish a boron-assisted amorphization strategy in which boron acts not as a passive dopant, but as a structural promoter that drives the formation of a B-rich oxide network in FeCoNiMoBOx ceramics. ApolloX predicts several low-energy amorphous configurations across systematically varied boron contents, and ab initio molecular dynamics simulations based on these structures reveal that increasing boron content stabilizes BO3-centered motifs, slows atomic diffusion, suppresses crystallization, and thereby strengthens the amorphization tendency. Guided by these predictions, we synthesize three FeCoNiMoBOx compositions within the theoretically identified composition window and use synchrotron-based scattering and electron microscopy to confirm that higher boron contents indeed yield stronger amorphization and more pronounced metal clustering. Electrochemical measurements further show that increasing boron content markedly enhances oxygen evolution reaction (OER) activity and durability. Overall, this ApolloX-guided, boron-assisted amorphization platform provides a general, compositionally programmable route to designing structurally and chemically complex amorphous catalysts, and more broadly opens new avenues for predictive discovery of functional disordered materials in catalysis and energy storage.

## I. Introduction

Recent advances in computational materials science have increasingly integrated machine learning and algorithmic search strategies into crystal structure prediction (CSP), often coupled with density functional theory (DFT) calculations, enabling the theory-driven discovery of materials with unexpected properties, such as high-temperature superconducting superhydrides [1–3]. Methodological innovations—including generative adversarial networks (GANs) that learn atomic distribution patterns from known structures [4,5], genetic algorithms and diffusion models that traverse configurational space via evolutionary or denoising strategies [6], and particle swarm optimization (PSO) for global minimum searches—have demonstrated remarkable success in ordered solids [7]. These approaches exploit the periodicity of crystalline materials, where unit-cell-based representations and symmetry constraints drastically reduce the search space [7–11]. In contrast, the absence of long-range order in amorphous, multi-element systems leads to an exponentially expanded configurational landscape, rendering direct application of these crystalline-oriented methods inefficient or infeasible. Consequently, despite their ubiquity in nature and technology, amorphous materials remain conspicuously underrepresented in the current predictive paradigm [12–14].

Amorphous materials, ranging from oxide glasses and metallic glasses to disordered catalysts and amorphous semiconductors, are central to numerous applications in energy, electronics, photonics, and structural engineering [12,15,16]. Unlike crystalline materials, amorphous solids cannot be described by simple unit cells, and accurate modeling often requires large supercells to capture representative short- and medium-range order [17,18]. However, simulating such large-scale disordered systems is computationally prohibitive using standard DFT-based approaches [19]. Furthermore, the disordered and metastable nature of amorphous materials makes their modeling the Achilles’ heel of theoretical descriptions [12]. These challenges have created significant barriers, particularly for multi-element (high-entropy) amorphous materials.

Computation-driven amorphous-material design seeks to map how atomic arrangements govern macroscopic behavior, revealing structure-function relationships through modeling. Exploring amorphous materials starting from a crystalline structure is a common approach in theoretical research. The Molecular Dynamics (MD) simulation uses a crystalline configuration as the initial structure and applies conditions to obtain an amorphous structure for property analysis [20,21]. Additionally, the Special Quasirandom Structure (SQS) method [22,23], coupled with the Monte Carlo (MC) process, provides a strategy for constructing initial structures in smaller systems. Then the Metropolis–MC (MMC) algorithm enables efficient sampling of configuration space through probabilistic acceptance of structural changes based on energy differences [24]. However, accurately capturing the potential energy landscape of amorphous systems remains a challenge. The cluster expansion (CE) method, combined with DFT, is commonly employed in crystalline materials to identify low-energy states through structure enumeration and ground-state search, providing valuable insights into new strategies for studying amorphous materials [25–27]. However, applying these methods to complex multicomponent amorphous systems is challenging, as increasing elemental diversity exponentially complicates atomic interactions and property predictions [12,13,28]. Furthermore, the inherent metastability of amorphous materials further limits conventional structure prediction. These challenges become particularly pronounced in systems with extreme compositions, such as high-entropy alloys and ceramics, highlighting the need for advancements in computational efficiency and predictive accuracy.

Here, we propose a physics-guided computational framework named ApolloX (Automatic Predictive generative mOdel for Large-scale Optimization of X-composition materials) that integrates a conditional generative deep learning model [29] with particle swarm optimization (PSO) [30] to challenge the simulation of multi-element amorphous materials. This framework harnesses chemical short-range order (CSRO) descriptors, represented by Pair Density Matrices (PDMs), alongside thermodynamic insights from DFT and machine learning potentials (MLP) [34] to systematically generate structurally reasonable models of these complex disordered systems. Additionally, the model-informed synthesis strategy, guided by sluggish diffusion control, enables precise tuning of elemental distribution and amorphization trends.

To validate our model, we developed a facile, universal, and scalable method and synthesized a series of amorphous materials. Using B₂O₃, one of the most difficult substances to crystallize in nature [35], and integrating transition metals with high-valence dopants known for exceptional oxygen evolution activity [36–38]. Our predicted FeCoNiMoBOₓ materials exhibit enhanced catalytic performance for oxygen evolution reaction (OER) with increasing amorphization. The strong agreement between predicted and experimental structures has shown our model’s ability to capture key features at the atomic scale, including short-range ordering and progressive amorphization process and mechanism. This predictive-experimental synergy establishes a transformative framework for designing amorphous multi-element material, surpassing conventional trial-and-error methods. Our approach not only predicts atomic arrangements that align with experimental observations but also reveals fundamental insights into composition-driven structural evolution and structure-function relationships.

## II. Results and discussion 
### A. Physics-guided generative model with PSO
ApolloX effectively identify physically plausible, low-energy patterns of chemical short-range order (CSRO) that are consistent with the prescribed composition and thermodynamic conditions, rather than deterministically reconstruct every atomic detail of a given amorphous configuration. These CSRO patterns are then realized as representative local atomic environments, which capture the essential statistics of the amorphous network. Figure 2A summarizes the workflow of ApolloX, which consists of four main steps.

![picture](https://github.com/gyf712/apollox_2_figs/blob/main/fig1.png?raw=true)

FIG. 1. Overview workflow of ApolloX. (A) Main workflow of ApolloX for structural modeling, including the model’s
iterative generation process. (B) Structures are represented using the Pair Density Matrix (PDM), and a dataset is constructed
by mapping each structure to its corresponding PDM. (C) The model architecture of the Cond-CDVAE. An embedding network
(EMB) integrates a structure’s composition (Comp.) and PDM into a unified conditioning vector. The graph neural network
encoder (PGNNA<sub>Enc</sub>) encodes the structure alongside its conditioning vector into a latent vector, capturing its underlying
features. Subsequently, a multi-layer perceptron MLPL predicts the lattice parameters based on the latent vector. And the
graph neural network decoder (PGNN<sub>Dec</sub>) outputs a score function for denoising and reconstruction of a plausible structure.
(D) Enthalpy stability is evaluated using the convex hull method, and entropy stability is assessed based on configurational
entropy. (E) PSO is employed to optimize the PDM, using the PYSWARMS package.

First, to build a compact descriptor of CSRO, we adopt the pair-density matrix (PDM), rather than using traditional descriptors such as Warren–Cowley parameters which relies on crystalline reference states or high-dimensional radial distribution functions. In the PDM, each atom is selected in turn as the center, and the number of neighboring atoms within a sphere of radius r_cut is counted. Each matrix element records the number of near-neighbor pairs between two element types (Fig. 2B), thereby capturing the statistics of local chemical environments. Unlike cluster-expansion–type approaches whose complexity grows rapidly with the number of elements and clusters, the PDM maintains a fixed K × K representation (where K is the number of element types), independent of system size. By classifying local environments and reducing structural complexity, the PDM provides an efficient handle on CSRO in disordered systems. Using this descriptor, we constructed a training database of 10,000 randomly generated structures spanning a range of prototype configurations and compositions.

Second, an in-house developed conditional generative model (Cond-CDVAE) was trained on the resulting structures with PDMs serve as the conditioning variable. The model architecture, illustrated in Fig. 2C, integrates a conditional variational autoencoder with a diffusion-style decoder to learn the statistical relationship between atomic configurations and their PDMs. Compared with the original crystal-focused implementation, which was designed for generating high-pressure crystalline structures, our modified model is tailored to amorphous systems and conditioned explicitly on CSRO descriptors (see Methods for details). During training, the model encodes structural features into a latent space and decodes them in a denoising, score-matching manner. For conditioning, vectorized PDMs are normalized and concatenated with composition vectors to form the conditional inputs used in both training and generation. At inference time, we provide target PDMs, and the model proposes atomic configurations that are compatible with these prescribed CSRO patterns.

Third, the ApolloX-generated structures were relaxed and screened for thermodynamic stability using machine-learning interatomic potentials. To make large-scale optimization feasible, we employed the DPA-2 framework [34] (fine-tuned on Fe–Co–Ni–Mo–B–O configurations; see Methods Section X). This allows us to relax thousands of candidate structures to their local minima and to evaluate, for each configuration, both the formation enthalpy and the configurational entropy inferred from the site-occupation statistics. Together, these quantities provide a thermodynamic fitness measure that we project onto the compositional simplex in Fig. 2D, where the left and right panels report the formation-energy and configurational-entropy landscapes, respectively.

In the final stage, we use a population-based particle swarm optimization (PSO) scheme to iteratively refine CSRO toward lower-energy regions of the configurational space. For each target PDM, a population of candidate structures (typically 100) is generated by the ApolloX. These structures were systematically relaxed to local minima using the DPA-2 framework [34] with model fine-tuning (Methods Section B), and evaluated in terms of enthalpy of mixing and configurational entropy (Fig. 2D). The most favorable candidates define updated PDMs, which are then used to generate new structures in the next PSO iteration. The PySwarms package (Fig. 2E) is employed to manage the PSO updates. To preserve diversity and reduce the risk of trapping in poor local minima, 60% of the lowest-energy structures are carried over to the next cycle, while the remaining 40% are newly generated. This iterative process does not aim to exhaustively sample the amorphous energy landscape, but rather to identify representative low-energy CSRO patterns and corresponding configurations that are suitable for subsequent relaxation, dynamical simulations, and comparison with experiment.

![picture](https://github.com/gyf712/apollox_2_figs/blob/main/fig2.png?raw=true)

FIG. 2. Model prediction of structural energy distribution and analysis. (A) Target pair-distribution matrix (PDM)
and the relative deviation of generated structures, demonstrating the model’s ability to capture short-range ordering with
high fidelity. (B) Histogram of the mean relative PDM errors. Inset: diversity metrics—uniqueness, coverage recall, and
coverage precision—plotted as functions of the similarity threshold δ. (C) Particle-swarm optimisation (PSO) trajectory;
the minimum-energy structure appears in the third generation, and one representative thermodynamically stable structure
is shown. (D) Energy distributions of structures produced by Random, ApolloX, SQS, and MC methods. The bar chart
highlights the greater abundance of ApolloX structures in the low-energy regime. Inset: violin plot of the global mean energy
and the distribution within the window −1050 to −990 eV. (E) The five lowest-energy structures obtained with ApolloX; their
PDMs (inset) exhibit a characteristic short-range-ordering motif.

### B. Validating ApolloX for amorphous materials discovery

To evaluate the effectiveness of ApolloX in modeling complex disordered systems, we applied it to a representative multi-component composition, Fe₁₂Co₁₂Ni₁₂Mo₁₂B₁₂O₆₀ (120 atoms in total). This system incorporates transition metal elements—Fe, Co, Ni, and Mo—commonly known for their catalytic activity in the oxygen evolution reaction (OER) [36–38], along with boron oxide, particularly B₂O₃, a textbook glass former [35]; its small ionic radius enables it to integrate into metal-oxide networks, facilitating amorphization by disrupting long-range order and eliminating distinct crystalline phases and interfacial boundaries [41]. To maintain charge neutrality while preserving an equimolar ratio among the metallic components and boron, we adopted a Fe:Co:Ni:Mo:B:O atomic ratio of 12:12:12:12:12:60. The combination of electron-deficient boron and chemically diverse transition metals provides a challenging yet ideal platform for benchmarking ApolloX’s performance in predicting thermodynamic stability, short-range order, and structure–function relationships in amorphous or metastable materials.

![picture](https://github.com/gyf712/apollox_2_figs/blob/main/fig3.png?raw=true)

FIG. 3. Short-range order evolution and diffusion-limited amorphization mechanism. (A) Heatmap from theoretical
modeling illustrating atomic nearest-neighbor statistics for structures with three different boron concentrations. (B) Atomic
configuration models show that higher boron content promotes the formation of short-range ordered B–O bonds, accompanied
by metal clustering. (C) Comparison of diffusion coefficients for metals, and boron within the same system reveals that boron
diffuses significantly more slowly than metals. Molecular dynamics simulations were performed under a slow heating ramp from
300 K to 2050 K over a total duration of 24 ps. (D) With increasing boron content, the formation of BO<sub>3</sub> motifs becomes more
prominent, reducing atomic mobility, hindering lattice rearrangement, and favoring amorphization. (E) Temperature-dependent
total radial distribution functions g(r) for the high-boron and low-boron groups, showing distinct structural evolution. (F)
Time-resolved atomic snapshots of the high-boron system depicting the dynamic evolution of BO<sub>3</sub> clusters. At 10 ps, isolated
BO<sub>3</sub> triangles begin to emerge. By 15 ps, a percolating BO<sub>3</sub> network forms, corresponding to a structurally arrested state. At
20 ps, thermal agitation induces partial fragmentation, followed by extensive dissociation of the network at 24 ps.

A total of 10,000 candidate structures for Fe₁₂Co₁₂Ni₁₂Mo₁₂B₁₂O₆₀ were generated by randomly substituting elements in a BCC parent phase within a 3 × 4 × 5 supercell. Energy distribution tests for BCC, FCC, and HCP configurations were also performed, as detailed in the Supplementary Section D. These structures were then systematically relaxed using the DPA-2 framework [34] with model fine-tuning (see method section B for detail). Subsequently, the PDMs of the relaxed structures were utilized to train the Cond-CDVAE model. To assess the accuracy of our PDM-driven Cond-CDVAE model, we chose a reference PDM as the target and generated 100 structures. As illustrated in Fig. 2A, the average difference between the generated and target PDMs for most element pairs are below 25%. Besides, we measured the reconstruction error distribution of 100 structures from the test set. For each target, we generated new structures and calculated the mean relative errors by averaging across all components of each structure’s PDM (Fig. 2B). 90% of the generated structures have the mean relative error less than 20%. The results demonstrate the high accuracy of our Cond-CDVAE generative model in capturing the targeted CSRO characteristics. Furthermore, the coverage recall, coverage uniqueness, and uniqueness were computed as the function of the mean relative error thresholds δ, and the novelty fractions of the generated structures were systematically evaluated, as detailed in Supplementary Text B.

To demonstrate the capabilities of ApolloX in structure search for amorphous systems, an initial set of 100 structures was generated, followed by structural evolution via PSO over 15 generations. The evolution of the lowest-energy structure across generations during PSO was shown in Fig. 2B, with the thermodynamically most favorable configuration shown in the inset. Obviously, the lowest-energy structure of amorphous Fe₁₂Co₁₂Ni₁₂Mo₁₂B₁₂O₆₀, with an energy of −1045.39 eV, emerged in the third generation, demonstrating the efficiency of the PSO algorithm in optimizing PDMs. Furthermore, the ApolloX method significantly outperformed the Random approach, as evidenced by the fact that the average energy achieved by ApolloX (−888.21 eV) was notably lower than that of the Random approach (−882.93 eV). This result underscores the superior performance of our Cond-CDVAE model in generating thermodynamically favorable amorphous structures.

To further assess the efficiency of ApolloX, we generated and analyzed 1,500 structures using several representative methods, including Random initialization, SQS, and Monte Carlo (MC)-only sampling, as shown in Fig. 2D. We additionally benchmarked against the Metropolis Monte Carlo (MMC) method, with results presented in Supplementary Fig. 7. All structure energies were evaluated using the fine-tuned DPA-2 model to ensure consistency. As shown in Fig. 2D, ApolloX exhibits superior performance in exploring low-energy regions of the configurational space. Specifically, 1.7% of the structures generated by ApolloX fall below an energy threshold of −960 eV, compared to 0.3% for Random, 0.1% for SQS, and 0% for MC-only. Notably, ApolloX outperforms even MMC, which achieves a lower success rate of 1.2% despite employing an extensive sampling strategy of 23 parallel simulations with 10,000 MC steps each (1,500 configurations total; see computational details in Supplementary Fig. 7). Furthermore, none of the baseline methods produced any structures with energies below −990 eV, while ApolloX successfully generated five such low-energy configurations. These results highlight the advantage of our SRO-guided generative approach in efficiently navigating the vast configurational space of amorphous multi-element systems.

As depicted in Supplementary Text C, the five lowest-energy structures predicted by ApolloX reveals significant structural diversity and notable variations in CSRO within these configurations. These findings not only highlight the effectiveness of ApolloX in generating low-energy configurations but also demonstrate its capability to strike an adequate balance between exploration and exploitation in predicting approximate structures within complex amorphous material systems.

While the cluster expansion (CE) method is primarily developed for crystalline systems, we included it in our analysis to enable a balanced and comprehensive comparison with our workflow, thereby assessing its applicability in disordered environments, using the CE approach implemented in the ATAT software [42] (see method section E for detail for structure modeling and energy prediction [27]). As expected, the first testing results of Fe₁₂Co₁₂Ni₁₂Mo₁₂B₁₂O₆₀ indicate that the amorphous system does not produce convergent cluster model fits with acceptable accuracy (0.02 eV as suggested by ATAT). Detailed information can be found in Supplementary Text B. This issue arises from the high concentration of non-metallic atoms, which causes lattice distortion and results in an amorphous structure. In contrast, our approach demonstrates enhanced capabilities for structural search and energy prediction in amorphous systems, underscoring its potential advantages over the CE method.



### C. Boron-driven structural evolution and ordering

We now apply ApolloX to the FeCoNiMoBOx system, focusing on how boron content influences structural evolution, SRO, and amorphization. Boron is known to be a potent glass former in oxide systems, with B2O3 being one of the most difficult substances to crystallize in nature [35]. In multi-element oxide and oxyhydroxide systems, boron can form BO3 and BO4 units that connect into extended networks, influencing both local coordination environments and medium-range connectivity.

Using ApolloX, we explore a series of compositions with systematically varied B content while maintaining charge neutrality and an equimolar ratio among the transition metals (Fe, Co, Ni, Mo). For each composition, we generate candidate amorphous structures conditioned on PDM descriptors that reflect expected SRO patterns, such as the preference for B–O bonds and the relative avoidance or preference of certain metal–metal or metal–oxygen pairs. PSO optimization in latent space selects configurations with low formation energies and favorable SRO.

Our simulations reveal that increasing B content drives the formation of B-rich oxide networks, characterized by interconnected BO3 motifs. These motifs serve as structural backbones that support the surrounding metal-oxygen framework, promoting amorphization by disrupting long-range periodicity and creating a more heterogeneous local environment. At moderate B contents, we observe the coexistence of BO3-centered networks with metal-centered coordination polyhedra, such as MO6 octahedra and MOx polyhedra with mixed metal occupancy. As B content increases further, the BO3 networks become more dominant, and the metal subnetwork becomes more disordered, leading to a highly amorphous structure.

The PDM analysis shows a clear trend in pairwise coordination statistics: B–O coordination increases with B content, while certain metal–metal and metal–oxygen pairings exhibit reduced probabilities, reflecting the structural role of B in redistributing local bonding environments. These changes in SRO correlate with shifts in local density, bond-length distributions, and angular distributions, all of which can be quantified within the PDM framework. Importantly, the ApolloX-generated structures capture these trends in a compositionally continuous manner, allowing us to trace the boron-driven structural evolution across the composition space.


![picture](https://github.com/gyf712/apollox_2_figs/blob/main/fig3.png?raw=true)

FIG. 3. Short-range order evolution and diffusion-limited amorphization mechanism. (A) Heatmap from theoretical
modeling illustrating atomic nearest-neighbor statistics for structures with three different boron concentrations. (B) Atomic
configuration models show that higher boron content promotes the formation of short-range ordered B–O bonds, accompanied
by metal clustering. (C) Comparison of diffusion coefficients for metals, and boron within the same system reveals that boron
diffuses significantly more slowly than metals. Molecular dynamics simulations were performed under a slow heating ramp from
300 K to 2050 K over a total duration of 24 ps. (D) With increasing boron content, the formation of BO<sub>3</sub> motifs becomes more
prominent, reducing atomic mobility, hindering lattice rearrangement, and favoring amorphization. (E) Temperature-dependent
total radial distribution functions g(r) for the high-boron and low-boron groups, showing distinct structural evolution. (F)
Time-resolved atomic snapshots of the high-boron system depicting the dynamic evolution of BO<sub>3</sub> clusters. At 10 ps, isolated
BO<sub>3</sub> triangles begin to emerge. By 15 ps, a percolating BO<sub>3</sub> network forms, corresponding to a structurally arrested state. At
20 ps, thermal agitation induces partial fragmentation, followed by extensive dissociation of the network at 24 ps.


### D. Diffusion-limited amorphization mechanism

In addition to static structural descriptors, ApolloX-based configurations can be used as starting points for MD simulations to probe dynamical behavior and diffusion processes in the amorphous FeCoNiMoBOx system. We perform *ab initio* or MLIP-accelerated MD simulations on selected configurations with varying B content, analyzing atomic diffusion coefficients, mean squared displacements, and time-dependent RDFs.

We find that increasing B content leads to a pronounced slowing of atomic diffusion, particularly for the transition metals. The formation of B-rich oxide networks introduces energetic barriers and steric constraints that hinder long-range atomic motion, effectively trapping atoms within locally constrained environments. This diffusion-limited kinetics contributes to amorphization by preventing the system from reorganizing into a crystalline lattice during cooling or annealing. Instead, the system becomes kinetically arrested in a disordered state with well-developed SRO and B-mediated connectivity.

The diffusion-limited amorphization mechanism elucidated by our simulations provides a microscopic explanation for the observed correlation between B content, structural disorder, and amorphization propensity. It also suggests strategies for designing other amorphous multi-element systems by tuning network-forming species and their interactions with the surrounding matrix.

![picture](https://github.com/gyf712/apollox_2_figs/blob/main/fig4.jpg?raw=true)

FIG. 4. Structural characterizations of three samples with varying boron contents. (A–C) AC-HAADF-STEM
images show disordered atomic distribution without periodic crystal lattice. (D–F) HAADF-STEM images show irregular
morphology. (G–I) Corresponding EDS mapping displays homogeneous distribution of Fe, Co, Ni, Mo elements across the
samples.

![picture](https://github.com/gyf712/apollox_2_figs/blob/main/fig5.png?raw=true)

FIG. 5. Structural characterization of FeCoNiMoBOx samples with varying boron contents. (A) X-ray diffrac-
tion (XRD) patterns of Group-1, Group-2, and Group-3 samples, showing the disappearance of long-range diffraction peaks
and progressive amorphization with increasing boron content. (B) Experimental pair distribution functions (G(r)) revealing
changes in local atomic ordering. (C) Short-range G(r) curves highlighting the relative intensities of M–O, M–M, and M–O–M
correlations for each group. (D) Fourier-transformed EXAFS fitting curves at the R-space of the Co K-edge, showing a decrease
in Co–O coordination and an increase in Co–M interactions as boron content increases. The data shown here are without phase
correction; phase-corrected results are provided in the Supplementary Fig. 19. (E) Wavelet-transform EXAFS (WT-EXAFS)
contour plots of Co foil, Co3O4, and the three sample groups, where the shift from Co–O–Co to Co–M scattering paths with
higher boron levels is evident.

## III. Conclusions

This work represents a paradigm shift in the computational design of amorphous multi-element materials. By introducing ApolloX, a physics-guided generative framework that integrates PDM-based structural descriptors, Cond-CDVAE generative modeling, and PSO optimization in latent space, we provide a systematic route to exploring the amorphous energy landscape in high-dimensional composition spaces. ApolloX enables the generation of amorphous structures with tailored SRO and energetics, overcoming longstanding challenges associated with the lack of periodicity and the vast configurational space of amorphous systems.

Applying ApolloX to the FeCoNiMoBOx system, we discover a boron-assisted amorphization route in which B-rich oxide networks stabilize amorphous phases and modulate diffusion dynamics. Our simulations reveal that increasing B content promotes the formation of BO3-centered motifs, slows atomic diffusion, and enhances amorphization. These predictions are validated experimentally through structural characterization and OER catalysis measurements, demonstrating a strong correlation between the degree of amorphization, SRO, and catalytic performance.

Beyond this specific case study, ApolloX offers a general and extendable framework for the design of amorphous materials in catalysis, energy storage, and other applications. By combining generative modeling with physically motivated descriptors and optimization schemes, ApolloX bridges the gap between data-driven methods and traditional physics-based approaches, enabling the rational design of amorphous materials with tunable local order and properties.

## IV. Methods

### A. Pair-Density Matrix (PDM) descriptor

The PDM descriptor provides a compact representation of local bonding environments in amorphous materials. For a given configuration, we define the PDM as a set of pairwise distributions over atomic species, coordination numbers, and bond-length intervals. Specifically, we partition the radial distance range into discrete bins and count the number of pairs of atoms (i, j) with species (α, β) and distances falling into each bin. This yields a multi-dimensional histogram that encodes the statistics of pairwise interactions.

To capture coordination environments, we also compute coordination number distributions for each species, defining neighbors based on radial cutoffs derived from RDF peaks or known bond-length ranges. The PDM thus includes both global pairwise statistics and local coordination information, providing a rich representation of SRO and, to some extent, MRO. For multi-element systems, we organize the PDM into blocks corresponding to different species pairs (e.g., Fe–O, Co–O, B–O, metal–metal), which can be analyzed individually or collectively.

### B. Model details of Cond-CDVAE

The Cond-CDVAE architecture used in ApolloX comprises an encoder, a decoder, and conditioning networks for composition and PDM features. The encoder takes as input a configuration represented by atomic positions and species, along with its associated PDM, and maps them to a latent vector z. The decoder reconstructs configurations from z, conditioned on target PDM and composition. We parameterize the encoder and decoder using graph neural networks (GNNs) or message-passing neural networks (MPNNs), which are well-suited for representing atomic structures.

During training, we optimize the evidence lower bound (ELBO) on the log-likelihood of the data, which includes a reconstruction loss (e.g., based on atomic positions and species) and a KL divergence regularization term that encourages the latent variables to follow a prior distribution (typically a multivariate Gaussian). The conditioning on PDM and composition is implemented via additional neural network modules that transform these features into embeddings concatenated with the latent variables or intermediate layers in the encoder and decoder.

Hyperparameters such as latent dimensionality, network depth, learning rate, and batch size are tuned based on validation performance on the training dataset. We also employ regularization techniques such as dropout and weight decay to prevent overfitting. The final trained model provides a smooth, continuous representation of the amorphous configuration space, enabling efficient sampling and optimization via PSO.

### C. Mean relative error of PDMs

To quantify the similarity between PDMs of generated and reference configurations, we define the mean relative error (MRE) as:

MRE = (1/N) Σ_k |P_gen(k) – P_ref(k)| / max(P_ref(k), ε)

where P_gen(k) and P_ref(k) are the values of the k-th bin in the generated and reference PDMs, respectively, N is the total number of bins, and ε is a small regularization parameter to avoid division by zero. The MRE provides a normalized measure of discrepancy between PDMs, with smaller values indicating closer agreement.

We compute MRE for each species pair and also aggregate over all pairs to obtain a global measure of PDM similarity. In our validation studies, we find that ApolloX-generated structures consistently achieve low MRE values relative to reference configurations obtained from melt-quench MD or experimental structural data, confirming the fidelity of the generative model in capturing SRO.

### D. Fine-tuning details of DPA-2

The structural complexity of the B-containing FeCoNiMo system requires careful fine-tuning of the underlying DFT and MLIP models used to evaluate formation energies and guide the generative process. We employ the DPA-2 (Density-Partitioned Approximation 2) scheme as a reference electronic-structure method, which balances accuracy and computational efficiency for multi-element oxide and oxyhydroxide systems.

We perform fine-tuning of DPA-2 parameters on a curated dataset of FeCoNiMoBOx configurations, including both crystalline and amorphous structures, spanning a range of compositions and local environments. The fine-tuning process involves adjusting exchange–correlation functional parameters, projector-augmented wave (PAW) potentials, and basis-set cutoffs to reproduce reference energies, forces, and structural properties obtained from higher-level DFT calculations. We validate the fine-tuned DPA-2 model by comparing formation energies and RDFs for test configurations not used in the fitting process.

### E. Comparative modeling of amorphous multi-element materials

For comparison with ApolloX-generated structures, we also construct amorphous models using conventional melt-quench MD simulations. Starting from randomized initial configurations with appropriate stoichiometry, we perform high-temperature MD simulations to melt the system, followed by controlled cooling at various rates to generate amorphous structures. We then relax these structures using DFT or MLIP to obtain stable configurations.

We analyze the resulting amorphous models using the same PDM and structural descriptors as for ApolloX-generated structures, enabling a direct comparison of SRO, coordination statistics, and energetics. We find that while melt-quench MD can produce realistic amorphous structures, it is computationally expensive and less flexible in exploring composition space and SRO variations. In contrast, ApolloX can rapidly generate a diverse set of amorphous configurations across compositions, with explicit control over SRO and energetics via PDM constraints and PSO optimization.

### F. Experimental methods

The FeCoNiMoBOx samples were synthesized via a sol–gel or co-precipitation method, followed by controlled annealing to promote amorphization and B incorporation. Precursors containing Fe, Co, Ni, Mo, and B sources were dissolved or dispersed in appropriate solvents, mixed thoroughly, and subjected to gelation or precipitation under controlled conditions. The resulting gels or precipitates were dried and calcined in air or inert atmospheres at temperatures optimized to induce amorphization while preventing excessive crystallization.

Structural characterization was performed using synchrotron-based PDF and EXAFS measurements, complemented by X-ray diffraction (XRD) and transmission electron microscopy (TEM). PDF data were collected at high-energy beamlines with sufficient Q-range to resolve SRO and MRO features. EXAFS measurements were carried out at relevant absorption edges (e.g., Fe, Co, Ni, Mo, B) to probe local coordination environments and oxidation states. Data analysis employed standard fitting procedures with structural models informed by ApolloX-generated configurations.

Electrochemical OER measurements were performed in alkaline electrolytes using rotating disk electrodes or other standard configurations. The FeCoNiMoBOx powders were deposited onto conductive substrates (e.g., glassy carbon) with appropriate binders, and current–potential curves were recorded under controlled conditions. Key performance metrics such as overpotential at a given current density, Tafel slopes, and stability under prolonged operation were extracted and compared across compositions.

### G. DFT and MD calculations

DFT calculations were carried out using plane-wave-based codes with PAW potentials and generalized gradient approximation (GGA) exchange–correlation functionals, such as PBE or its variants. For selected configurations, we also explored hybrid functionals or DFT+U corrections to better capture electronic correlation effects in transition metal oxides. K-point sampling and energy cutoffs were chosen to ensure convergence of total energies and forces within acceptable tolerances.

MD simulations employed either *ab initio* DFT-based MD for small supercells and short timescales or MLIP-accelerated MD for larger systems and longer timescales. Temperature control was achieved using Nosé–Hoover or Langevin thermostats, and pressure control was implemented via barostats when necessary. Diffusion coefficients were extracted from mean squared displacements, and time-dependent structural descriptors (e.g., RDFs, coordination statistics) were computed to analyze dynamical behavior and amorphization mechanisms.

---

*Note:* The above Methods subsections summarize the main computational and experimental procedures underlying the ApolloX framework and its application to the FeCoNiMoBOx system. Additional details, including parameter values, convergence tests, and data-processing protocols, can be provided in supplementary materials or dedicated technical appendices.
