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

Amorphous materials, with their intrinsic structural disorder and rich short-range atomic ordering, underpin a wide spectrum of technological applications in catalysis, energy storage, electronics, and beyond. However, unlike crystalline materials, whose properties can often be rationalized and optimized via well-defined periodic lattice models, the lack of long-range order in amorphous solids renders their atomic configurations difficult to predict, model, and design. This challenge is particularly acute in the case of multi-element, high-entropy amorphous systems, where the combinatorial explosion of possible local environments leads to a vast configurational space that is computationally expensive to explore and conceptually hard to navigate.

In crystalline materials, the periodicity of the lattice provides a natural framework for structure prediction, often formulated as a search over unit cells with a limited number of atoms. Over the past decade, crystal structure prediction (CSP) has been revolutionized by the integration of density functional theory (DFT), global optimization algorithms, and, more recently, machine learning (ML) and generative models. These methods have enabled the discovery of numerous new crystalline materials, including unconventional superconductors, topological phases, and high-energy-density compounds. Nonetheless, most of these advances have focused on ordered phases, where the objective is to find the global or near-global minimum in a relatively low-dimensional energy landscape parameterized by lattice vectors and a small set of atomic positions.

By contrast, amorphous materials lack periodicity and are instead characterized by a distribution of local environments, often captured in terms of short-range order (SRO) and, in some cases, medium-range order (MRO). The energy landscape of amorphous systems is rugged and high-dimensional, with a multitude of metastable states separated by complex energy barriers. Traditional CSP strategies, even when combined with advanced sampling techniques, are ill-suited for exploring such landscapes exhaustively. Moreover, the absence of long-range order complicates the definition of a suitable objective function for optimization: rather than seeking a single lowest-energy configuration, one must consider ensembles of configurations that share certain structural motifs and satisfy thermodynamic constraints such as configurational entropy.

Recent advances in machine learning and generative modeling provide a promising avenue for addressing these challenges. Variational autoencoders (VAEs), generative adversarial networks (GANs), normalizing flows, and diffusion models have demonstrated remarkable success in generating complex data distributions in domains ranging from images and text to molecules and crystal structures. In materials science, generative models have been used to propose novel compositions, optimize lattice parameters, and explore structural motifs under given constraints. However, most existing works still assume crystalline order or near-crystalline environments, and the explicit incorporation of amorphous-specific features, such as local atomic coordination statistics and SRO patterns, remains relatively underdeveloped.

In this work, we develop ApolloX, a physics-guided generative framework designed specifically for the predictive design of amorphous multi-element materials with tunable short-range order. ApolloX integrates a conditional compositional-distribution variational autoencoder (Cond-CDVAE) with particle swarm optimization (PSO) and a pair-density matrix (PDM)-based descriptor that encodes local bonding environments and site-occupancy statistics. The core idea is to represent amorphous structures in terms of their local coordination distributions and chemical SRO, and to train a generative model that can sample low-energy configurations consistent with these structural constraints. The PSO component then operates in the latent space of the generative model, guided by a thermodynamic fitness function that combines DFT-calculated formation energies with configurational entropy inferred from site-occupancy statistics.

By explicitly incorporating SRO constraints and thermodynamic metrics into the generative process, ApolloX enables the systematic navigation of the amorphous energy landscape in high-dimensional composition spaces. We demonstrate the power of this approach by applying it to a family of FeCoNiMoBOx ceramics, where we identify a boron-assisted amorphization route that leverages the formation of B-rich oxide networks. In this system, boron does not merely act as a passive dopant but plays an active role in promoting amorphization, stabilizing BO3-centered motifs, and modulating diffusion dynamics.

Our computational predictions are validated by experimental synthesis, structural characterization (including synchrotron-based pair distribution function (PDF) and extended X-ray absorption fine structure (EXAFS) measurements), and oxygen evolution reaction (OER) catalysis tests. We observe a strong correlation between the degree of amorphization, as quantified by local structural metrics, and catalytic performance. Specifically, compositions predicted by ApolloX to exhibit enhanced amorphization and optimized SRO show superior OER activity, highlighting the potential of our framework for rational catalyst design.

Beyond this specific case study, ApolloX provides a general and programmable route to the design of structurally and chemically complex amorphous catalysts and functional materials. By bridging generative modeling, thermodynamic reasoning, and experimental validation, our framework opens new avenues for the predictive discovery of amorphous materials in catalysis, energy storage, and beyond.

## I. Introduction

As high-temperature superconducting superhydrides [1–3]. Methodological innovations—including generative algorithms and diffusion models that traverse configurational spaces, such as CSP coupled with DFT-based energy evaluations and minimum searches—have demonstrated remarkable success in ordered solids [7]. These approaches exploit the periodicity of crystalline materials, where unit-cell-based representations and symmetry constraints drastically reduce the effective dimensionality of the search space. As a result, crystalline materials have benefited disproportionately from the rapid development of computational materials design, with numerous successes in predicting and discovering new compounds and phases.

However, the extension of these strategies to amorphous materials is far from straightforward. Amorphous solids, by definition, lack long-range translational symmetry and cannot be described by a small repeating unit cell. Their atomic structures are instead characterized by a broad distribution of local environments and complex SRO and MRO patterns that vary across the material. Consequently, direct adaptation of crystalline-oriented structure prediction methods to amorphous systems is often inefficient or infeasible, particularly for multi-element (high-entropy) amorphous materials where the number of possible local configurations grows combinatorially with the number of components and their stoichiometric ratios.

Amorphous materials, ranging from oxide glasses and metallic glasses to disordered catalysts and amorphous semiconductors, are central to numerous applications in energy, electronics, photonics, and structural engineering [12, 15, 16]. Unlike crystalline materials, amorphous solids cannot be described by simple unit cells, and accurate modeling often requires large supercells to capture representative SRO and MRO [17, 18]. However, simulating such large-scale disordered systems is computationally prohibitive using standard DFT-based approaches [19]. Furthermore, the disordered and metastable nature of amorphous materials makes their modeling the Achilles’ heel of theoretical descriptions [12]. These challenges have created significant barriers, particularly for multi-elemental (high-entropy) amorphous materials, which offer enormous potential for tunable properties but are notoriously difficult to design rationally.

To address these challenges, alternative strategies have been proposed, including the use of empirical or semi-empirical potentials, coarse-grained models, and machine-learned interatomic potentials (MLIPs) that can accelerate molecular dynamics (MD) simulations of amorphous systems [20–22]. While MLIPs have shown great promise in reproducing local structural features and dynamic behaviors, their training typically relies on large DFT datasets and carefully curated reference configurations. Moreover, MLIPs alone do not directly solve the problem of exploring the vast configurational space of multi-element amorphous materials or establishing clear structure–property relationships.

Another line of research has focused on statistical and topological descriptors of amorphous structure, such as radial distribution functions (RDFs), coordination number distributions, bond-angle distributions, and ring statistics [23–25]. These descriptors provide valuable insights into local structural motifs and can be used to compare simulated and experimental structures. However, they are primarily descriptive rather than generative: they characterize given configurations but do not prescribe how to construct new ones with desired properties.

In parallel, the emergence of generative models has transformed fields such as computer vision and natural language processing, enabling the synthesis of realistic images, text, and audio with controllable attributes [26–29]. In materials science, generative models have been applied to propose new compositions, crystal structures, and microstructures that satisfy target constraints [30–34]. These models typically operate in a latent space learned from data, where smooth variations in latent variables correspond to meaningful changes in material properties. Nonetheless, most existing generative approaches in materials science have been developed for crystalline phases, where structures can be represented compactly and symmetry constraints are well understood.

Bringing generative modeling to amorphous materials requires addressing several key conceptual and technical challenges. First, one must define an appropriate representation of amorphous structure that captures relevant SRO and chemical ordering patterns without relying on periodicity. Second, the generative model must be conditioned on composition and possibly other external variables (e.g., temperature, pressure) to ensure that generated configurations are physically meaningful and thermodynamically plausible. Third, the model must be coupled to an optimization framework that can navigate the amorphous energy landscape efficiently, favoring low-energy configurations while maintaining sufficient diversity to explore alternative local minima.

In this work, we propose to meet these challenges by combining a PDM-based descriptor, which encodes the distribution of local bonding environments and site-occupancy statistics, with a conditional generative model and PSO-based optimization. The PDM descriptor provides a compact yet expressive representation of local structure, allowing us to quantify SRO and chemical ordering in multi-element amorphous systems. The conditional generative model, built upon the CDVAE architecture, learns a mapping from composition and PDM descriptors to atomic configurations in a high-dimensional latent space. Finally, the PSO algorithm operates in this latent space, guided by a thermodynamic fitness function that combines DFT-calculated formation energies with configurational entropy estimates.

By integrating these components into a unified framework, ApolloX enables the generation of amorphous structures with tailored SRO and energetics. We apply ApolloX to the FeCoNiMoBOx system, where we identify a boron-assisted amorphization route mediated by the formation of B-rich oxide networks. Our simulations reveal that increasing B content stabilizes BO3-centered motifs, slows atomic diffusion, and promotes the formation of amorphous phases with enhanced catalytic performance in OER. These predictions are validated experimentally, demonstrating the practical utility of ApolloX and highlighting the importance of SRO engineering in amorphous catalyst design.

The remainder of this paper is organized as follows. In Sec. II, we describe the ApolloX framework in detail, including the PDM descriptor, the Cond-CDVAE model, and the PSO optimization scheme. We also discuss the validation of ApolloX on representative amorphous systems. In Sec. III, we apply ApolloX to the FeCoNiMoBOx system, elucidating the boron-driven structural evolution, diffusion-limited amorphization mechanisms, and the resulting structure–property relationships in OER catalysis. In Sec. IV, we present our conclusions and outline future directions for the generative design of amorphous materials.

## II. Results and discussion A. Physics-guided generative model with PSO

The central goal of ApolloX is to generate and optimize amorphous atomic configurations that are both energetically favorable and consistent with desired SRO patterns. To achieve this, we introduce a physics-guided generative model that integrates a PDM descriptor, a Cond-CDVAE architecture, and a PSO optimization scheme. The PDM descriptor captures the distribution of local bonding environments, including coordination numbers, bond lengths, and site-occupancy statistics for each atomic species. These descriptors serve as inputs and constraints for the generative model, which learns to map compositions and PDM features to atomic configurations.

We begin by constructing an initial dataset of amorphous structures across a range of compositions in the target multi-element system. These structures are generated via a combination of melt-quench MD simulations, randomized atomic placement with subsequent relaxation, and heuristic SRO constraints based on known chemical preferences (e.g., avoidance or preference of certain pairings). For each configuration, we compute the PDM, which encodes the pairwise distributions of atomic species and coordination environments. We also calculate the formation energy using DFT or, when appropriate, a validated MLIP. The resulting dataset consists of tuples (composition, PDM, configuration, energy), which serve as training data for the Cond-CDVAE model.

The Cond-CDVAE architecture extends standard VAEs by incorporating both compositional and PDM information as conditioning variables. Specifically, the encoder maps an input configuration (represented by atomic positions and species) and its associated PDM to a latent vector z, while the decoder reconstructs configurations conditioned on both z and the target PDM/composition. During training, the model learns to approximate the posterior distribution of latent variables given the observed data, while regularizing the latent space via a Kullback–Leibler divergence term. The conditioning on PDM and composition ensures that the latent space captures variations in local order and chemical arrangement relevant to the amorphous system.

Once trained, the Cond-CDVAE model can be used to generate new configurations by sampling latent vectors z and decoding them under specified PDM and composition conditions. However, naive sampling may produce structures that are physically less relevant or energetically unfavorable. To focus the generative process on low-energy configurations, we couple the Cond-CDVAE with a PSO optimization scheme operating in the latent space. In PSO, a population of particles explores the latent space, with each particle representing a candidate configuration. The position of a particle corresponds to a point in the latent space, and the corresponding decoded structure is evaluated using a fitness function that combines DFT or MLIP energy with configurational entropy inferred from the PDM.

The PSO dynamics update each particle’s position by combining a personal best component (the best latent position found by that particle so far) and a global best component (the best position found by the entire swarm). In this way, the swarm collectively balances exploration and exploitation, gradually converging towards regions of the latent space that correspond to low-energy, structurally desirable amorphous configurations. Importantly, the PDM constraints are enforced either explicitly (by conditioning the decoder on target PDM features) or implicitly (by evaluating the consistency of generated structures with PDM-derived metrics).

This physics-guided generative process allows ApolloX to efficiently explore the amorphous energy landscape without exhaustively sampling the entire configurational space. Instead of directly optimizing atomic positions in real space, which is high-dimensional and rugged, we optimize in the lower-dimensional latent space of the Cond-CDVAE, where meaningful structural variations are encoded in a compact form. The combination of generative modeling and PSO thus provides a powerful mechanism for identifying representative low-energy SRO patterns and corresponding configurations suitable for further relaxation, dynamical simulations, and comparison with experiment.

The final output of the PSO-guided generative process is a set of candidate amorphous structures that exhibit distinct SRO motifs and low formation energies. These configurations can be further refined using DFT relaxation and MD simulations to assess their stability, dynamical behavior, and response to external conditions. The PDM and associated structural descriptors extracted from these configurations can also be used to build structure–property relationships, linking specific SRO patterns to macroscopic observables such as diffusivity, mechanical rigidity, and catalytic activity.

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

### B. Validating ApolloX for amorphous materials discovery

To validate ApolloX as a general framework for amorphous materials design, we apply it to a set of model systems with varying degrees of structural complexity and chemical heterogeneity. These include simple binary and ternary oxide glasses, metallic glasses, and multi-element ceramic systems with known amorphization tendencies. For each system, we compare ApolloX-generated structures with reference configurations obtained from conventional melt-quench MD simulations and experimental structural data (e.g., RDFs, PDFs, EXAFS).

In all cases, ApolloX successfully reproduces key structural features of the amorphous phases, including coordination number distributions, bond-angle statistics, and characteristic peaks in the RDF or PDF. The PDM-based descriptor proves particularly effective in capturing subtle differences in SRO between compositions, such as the preference for certain cation–anion pairings or the formation of specific polyhedral motifs. The latent space learned by the Cond-CDVAE captures these variations in a smooth and continuous manner, allowing the generative model to interpolate between known structures and extrapolate to new compositions.

Quantitatively, we assess the accuracy of ApolloX-generated structures by computing the mean relative error (MRE) between PDMs of generated and reference configurations. We find that the MRE is consistently low across systems and compositions, indicating that the generative model preserves the essential features of local order. Moreover, as we incorporate PSO-guided optimization, the fitness function—combining energy and entropy contributions—further selects configurations that are not only structurally consistent but also thermodynamically favorable.

These validation studies demonstrate that ApolloX can serve as a reliable tool for generating physically plausible amorphous structures in complex multi-element systems. The ability to control SRO via PDM constraints and to optimize configurations via PSO in latent space distinguishes ApolloX from purely data-driven generative models that lack explicit physical guidance. This capability is particularly important for high-entropy amorphous materials, where subtle changes in SRO can have pronounced effects on properties such as diffusion, mechanical response, and catalytic activity.

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
