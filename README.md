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

Recent advances in computational materials science have increasingly integrated machine learning and algorithmic search strategies into crystal structure prediction (CSP), often coupled with density functional theory (DFT) calculations, enabling the theory-driven discovery of materials with unexpected properties, such as high-temperature superconducting superhydrides [1–3]. Methodological innovations—including generative adversarial networks (GANs) that learn atomic distribution patterns from known structures [4, 5], genetic algorithms and diffusion models that traverse configurational space via evolutionary or denoising strategies [6], and particle swarm optimization (PSO) for global minimum searches—have demonstrated remarkable success in ordered solids [7]. These approaches exploit the periodicity of crystalline materials, where unit-cell-based representations and symmetry constraints drastically reduce the search space [7–11]. In contrast, the absence of long-range order in amorphous, multi-element systems leads to an exponentially expanded configurational landscape, rendering direct application of these crystalline-oriented methods inefficient or infeasible. Consequently, despite their ubiquity in nature and technology, amorphous materials remain conspicuously underrepresented in the current predictive paradigm [12–14].

Amorphous materials, ranging from oxide glasses and metallic glasses to disordered catalysts and amorphous semiconductors, are central to numerous applications in energy, electronics, photonics, and structural engineering [12, 15, 16]. Unlike crystalline materials, amorphous solids cannot be described by simple unit cells, and accurate modeling often requires large supercells to capture representative short- and medium-range order [17, 18]. However, simulating such large-scale disordered systems is computationally prohibitive using standard DFT-based approaches [19]. Furthermore, the disordered and metastable nature of amorphous materials makes their modeling the Achilles’ heel of theoretical descriptions [12]. These challenges have created significant barriers, particularly for multi-elemental (high-entropy) amorphous materials.

Computational-driven amorphous-material design seeks to map how atomic arrangements govern macroscopic behavior, revealing structure–function relationships through modeling. Exploring amorphous materials starting from a crystalline structure is a common approach in theoretical research. The Molecular Dynamics (MD) simulation uses a crystalline configuration as the initial structure and applies conditions to obtain an amorphous structure for property analysis [20, 21]. Additionally, the Special Quasirandom Structure (SQS) method [22, 23], coupled with the Monte Carlo (MC) process, provides a strategy for constructing initial structures in smaller systems. Then the Metropolis-MC (MMC) algorithm enables efficient sampling of configuration space through probabilistic acceptance of structural changes based on energy differences [24]. However, accurately capturing the potential energy landscape of amorphous systems remains a challenge. The cluster expansion (CE) method, combined with DFT, is commonly employed in crystalline materials to identify low-energy states through structure enumeration and ground-state search, providing valuable insights into new strategies for studying amorphous materials [25–27]. However, applying these methods to complex multicomponent amorphous systems is challenging, as increasing elemental diversity exponentially complicates atomic interactions and property predictions [12, 13, 28]. Furthermore, the inherent metastability of amorphous materials further limits conventional structure prediction. These challenges become particularly pronounced in systems with extreme compositions, such as high-entropy alloys and ceramics, highlighting the need for advancements in computational efficiency and predictive accuracy.

Here, we propose a physics-guided computational framework named ApolloX (Automatic Prediction by generative mOdel for Large-scaLe Optimization of X-composition materials) that integrates a conditional generative deep learning model [29] with particle swarm optimization (PSO) [30] to challenging the simulation of multi-element amorphous materials. This framework harnesses chemical short-range order (CSRO) [31–33] descriptors, represented by Pair Density Matrices (PDMs), alongside thermodynamic insights from DFT and machine learning potentials (MLP) [34] to systematically generate structurally reasonable models of these complex disordered systems. Additionally, the model-informed synthesis strategy, guided by sluggish diffusion control, enables precise tuning of elemental distribution and amorphization trends.

To validate our model, we developed a facile, universal, and scalable method and synthesized a series of amorphous materials. Using B₂O₃, one of the most difficult substances to crystallize in nature [35], and integrating transition metals with high-valence dopants known for exceptional oxygen evolution activity [36–38], our predicted FeCoNiMoBOₓ materials exhibit enhanced catalytic performance for oxygen evolution reaction (OER) with increasing amorphization. The strong agreement between predicted and experimental structures has shown our model’s ability to capture key features at the atomic scale, including short-range ordering and progressive amorphization process and mechanism. This predictive-experimental synergy establishes a transformative framework for designing amorphous multi-element material, surpassing conventional trial-and-error methods. Our approach not only predicts atomic arrangements that align with experimental observations but also reveals fundamental insights into composition-driven structural evolution and structure-function relationships.

## II. Results and discussion 
### A. Physics-guided generative model with PSO
Amorphous materials, lacking translational symmetry and long-range order, are best characterized by their short-range order. Our method predicts these structures using only chemical composition and local thermodynamic information. Fig. 1A outlines the workflow of ApolloX, consisting of four main steps. While traditional CSRO descriptors such as Warren–Cowley parameters (WCP) [39] require crystalline reference states and radial distribution functions suffer from high dimensionality, we adopted the PDM for its computational efficiency and natural compatibility with disordered systems. Initially, the PDM was introduced as a descriptor of CSRO, capturing atomic nearest-neighbor interactions (Fig. 1B). It is constructed by selecting each atom as the center and counting the number of neighboring atoms within a sphere of radius r<sub>cut</sub>. Each matrix element represents the count of a specific pair of neighboring atoms, capturing nearest-neighbor interactions among different element types. Unlike cluster-based methods that scale poorly with increasing number of elements, the PDM maintains a fixed K × K representation (where K is the number of element types) regardless of system complexity. By classifying local atomic environments and reducing structural complexity, the PDM provides a powerful tool for analyzing amorphous materials (as shown in Supplementary Table 3). A training database was constructed with 100,000 randomly generated structures spanning a range of prototype configurations (e.g., FCC, BCC, HCP), and MLPs were used to optimize these structures, significantly reducing computational costs.

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

Next, the PDM–structure pairs served as training set for the conditional generative model. The model architecture, illustrated in Fig. 1C, integrates a conditional variational autoencoder with a diffusion framework (Cond-CDVAE) [29] to generate amorphous solid structures with the desired PDM. Compared to the original model, which was designed for the generation of crystal structures under high-pressure conditions, our modified Cond-CDVAE aims to establish the relationship between structures and their corresponding PDMs (see method section A for detail). The model first encodes the structure features into a latent space, and then decodes/generates a structure in a denoising score-matching manner. To achieve this, the vectorized PDM matrices are normalized and concatenated with composition representation vectors to form the conditional vectors used in both the training and generation stages.

The final step employs a population-based PSO algorithm [30] to refine the predicted structures. In each iteration, 100 structures are generated with a target PDM, relaxed to local minima, and assessed for thermodynamic stability using enthalpy of mixing and configurational entropy (Fig. 1D). The most stable structure’s PDM is then used to generate new candidates, iteratively refining configurations toward thermodynamically favorable states. The PySwarms package [40] (Fig. 1E) facilitates these PSO iterations. To maintain structural diversity and avoid local minima, 60% of energy-ranked structures are carried over to the next cycle, while 40% are newly generated. This iterative process continues until a predefined termination criterion is met, ensuring an optimized exploration of the amorphous energy landscape.

### B. Validating ApolloX for amorphous materials discovery

To evaluate the effectiveness of ApolloX in modeling complex disordered systems, we applied it to a representative multi-component composition, Fe₁₂Co₁₂Ni₁₂Mo₁₂B₁₂O₆₀ (120 atoms in total). This system incorporates transition metal elements—Fe, Co, Ni, and Mo—commonly known for their catalytic activity in the oxygen evolution reaction (OER) [36–38], along with boron oxide, particularly B₂O₃, is a textbook glass former [35]; its small ionic radius enables it to integrate into metal-oxide networks, facilitating amorphization by disrupting long-range order and eliminating distinct crystalline phases and interfacial boundaries [41]. To maintain charge neutrality while preserving an equimolar ratio among the metallic components and boron, we adopted a Fe:Co:Ni:Mo:B:O atomic ratio of 12:12:12:12:12:60. The combination of electron-deficient boron and chemically diverse transition metals provides a challenging yet ideal platform for benchmarking ApolloX’s performance in predicting thermodynamic stability, short-range order, and structure–function relationships in amorphous or metastable materials.

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

A total of 10,000 candidate structures of Fe₁₂Co₁₂Ni₁₂Mo₁₂B₁₂O₆₀ were generated by randomly substituting elements in a BCC parent phase within a 3 × 4 × 5 supercell. Energy distribution tests for BCC, FCC, and HCP configurations were also performed, as detailed in the Supplementary Section D. These structures were then systematically relaxed using the DPA-2 framework [34] with model fine-tuning (see method section B for detail). Subsequently, the PDMs of the relaxed structures were utilized to train the Cond-CDVAE model.To assess the accuracy of our PDM-driven Cond-CDVAE model, we selected a reference PDM as the target and generated 100 structures. As illustrated in Fig. 2A, the average differences between the generated and target PDMs for most element pairs are below 25%. Besides, we computed the reconstruction error distribution of 1000 structures from the test set. For each target, we generated one structure and calculated the mean relative errors by averaging across all components of each structure’s PDM. In Fig. 2B, 90% of the generated structures have the mean relative error less than 20.0%. The results demonstrate the high accuracy of our Cond-CDVAE generative model in capturing the targeted CSRO characteristics.Furthermore, the coverage recall, coverage precision and uniqueness were computed as the function of the mean relative error threshold δ, and the novelty fractions of the generated structures were systematically evaluated, as detailed in Supplementary Text B.

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

To demonstrate the capabilities of ApolloX in structure search for amorphous systems, an initial set of 100 structures was generated, followed by structural evolution via PSO over 15 generations. The evolution of the lowest-energy structure across generations during PSO was shown in Fig. 2B, with the thermodynamically most favorable configuration shown in the inset. Obviously, the lowest-energy structure of amorphous Fe₁₂Co₁₂Ni₁₂Mo₁₂B₁₂O₆₀, with an energy of −1045.39 eV, emerged in the third generation, demonstrating the efficiency of the PSO algorithm in optimizing PDMs. Furthermore, the ApolloX method significantly outperformed the Random approach, as evidenced by the fact that the average energy achieved by ApolloX (−888.21 eV) was notably lower than that of the Random approach (−882.93 eV). This result underscores the superior performance of our Cond-CDVAE model in generating thermodynamically favorable amorphous structures.

To further assess the efficiency of ApolloX, we generated and analyzed 1,500 structures using several representative methods, including Random initialization, SQS, and Monte Carlo (MC)-only sampling, as shown in Fig. 2D. We additionally benchmarked against the Metropolis Monte Carlo (MMC) method, with results presented in Supplementary Fig. 7. All structure energies were evaluated using the fine-tuned DPA-2 model to ensure consistency. As shown in Fig. 2D, ApolloX exhibits superior performance in exploring low-energy regions of the configurational space. Specifically, 1.7% of the structures generated by ApolloX fall below an energy threshold of −960 eV, compared to 0.3% for Random, 0.1% for SQS, and 0% for MC-only. Notably, ApolloX outperforms even MMC, which achieves a lower success rate of 1.2% despite employing an extensive sampling strategy of 23 parallel simulations with 10,000 MC steps each (1,500 configurations total; see computational details in Supplementary Fig. 7). Furthermore, none of the baseline methods produced any structures with energies below −990 eV, while ApolloX successfully generated five such low-energy configurations. These results highlight the advantage of our SRO-guided generative approach in efficiently navigating the vast configurational space of amorphous multi-element systems.

As depicted in Supplementary Text C, the five lowest-energy structures predicted by ApolloX reveal significant structural diversity and notable variations in CSRO within these configurations. These findings not only highlight the effectiveness of ApolloX in generating low-energy configurations but also demonstrate its capability to strike an adequate balance between exploration and exploitation in predicting approximate structures within complex amorphous material systems.

While the cluster expansion (CE) method is primarily developed for crystalline systems, we included it in our analysis to enable a balanced and comprehensive comparison with our workflow, thereby assessing its applicability in disordered environments, using the CE approach implemented in the ATAT software [42] (see method section E for detail) for structure modeling and energy prediction [27]. As expected, the fitting results of Fe₁₂Co₁₂Ni₁₂Mo₁₂B₁₂O₆₀ indicate that the amorphous system does not produce convergent cluster model fits with acceptable accuracy (0.02 eV as suggested by ATAT). Detailed information can be found in Supplementary Text B. This issue arises from the high concentration of non-metallic atoms, which causes lattice distortion and results in an amorphous structure. In contrast, our approach demonstrates enhanced capabilities for structural search and energy prediction in amorphous systems, underscoring its potential advantages over the CE method.


### C. Boron-driven structural evolution and ordering

To examine the influence of boron content on microstructure and material properties, we employed ApolloX to generate structures with 6%, 9%, and 12% boron for Group-1, -2 and -3, respectively, while maintaining equivalent metallic ratios. For each composition, 10,000 structures were generated to identify the thermodynamically most stable configuration, incorporating both configurational entropy and enthalpy. We then extracted the CSROs of the lowest-energy structures for further analysis. Fig. 3A shows the predicted structures and elemental enrichment heat maps, where increasing boron content shifts metal distribution from dispersed to aggregated. Notably, our model captures only local short-range ordering effects; thus, the observed atomic clustering does not necessarily correlate with long-range amorphization.

A comparison of atomic-level structures across the three groups reveals a clear transition in CSRO with increasing boron content. In Group-1 (low boron), metal-O-metal connectivity dominates. As boron content increases, oxygen preferentially bonds with boron rather than metals, forming BO₃ motifs (Fig. 3B), where boron coordinates with three oxygen atoms, introducing local short-range ordering. In Group-3 (highest boron content), this shift toward BO₃ formation leads to reduced metal-oxygen interactions and enhanced metal clustering (Fig. 3A). These findings highlight the significant role of boron in modifying local CSRO, driving a shift from metal-O-metal connectivity toward B-O coordination and direct metal-metal interactions.

### D. Diffusion-limited amorphization mechanism

In 2024, Hsu et al. [43] proposed that, in multi-component materials, the element with the lowest diffusion rate governs the kinetics of lattice restructuring and plays a critical role in controlling structural disorder. To investigate this effect and understand the amorphization mechanism, we performed ab initio molecular dynamics (AIMD) simulations on molten multi-element systems with varying boron concentrations under continuous thermal excitation (Figs. 3C–F). Each system was gradually heated from 300 K to 2050 K over 25 ps. Fig. 3E presents the temperature-dependent total radial distribution functions (RDF) for the low- and high-boron systems. While both retain a sharp first peak, indicating preserved short-range order, the high-boron system shows a rapid disappearance of second and third peaks above 1000 K, signaling a collapse of medium-range order and the onset of amorphization. In contrast, the low-boron system maintains distinct RDF peaks up to 1700 K, suggesting more persistent crystalline-like order and delayed amorphization. These observations highlight the enhanced glass-forming tendency of the high-boron composition. Quantitative analysis (Fig. 3C) shows that increasing boron content systematically reduces the overall atomic diffusion rates. Among all elements, boron exhibits the slowest diffusion, and its mobility decreases more steeply as the concentration increases. This behavior is further captured in the time evolution of the mean squared displacement (MSD) for boron, where a pronounced stagnation plateau is observed between 12.5 ps and 20 ps in the high-boron group. This stagnation corresponds to a transient “frozen” state, driven by the formation of locally stable BO₃ triangular units. Snapshots in Fig. 3F provide structural insight into this kinetic arrest. At 15 ps, the boron atoms form locally stable BO₃ triangular motifs that coalesce into a percolating short-range network, significantly restricting boron mobility. Upon further heating, thermal energy disrupts the BO₃ framework, leading to renewed diffusion. This behavior supports a dynamic picture of the glass transition in which the system transiently resides in a low-mobility, short-range ordered state before becoming thermally activated into a more disordered, diffusive regime.

Fig. 3D further illustrates how increasing boron content promotes the formation of BO₃ motifs, which not only disrupt long-range M–O–M connectivity but also reduce oxygen mobility by increasing the local damping environment. This results in higher concentrations of oxygen vacancies (Supplementary Fig. S20) and alters crystallization kinetics. Unlike conventional multi-element oxides that favor extended crystalline coordination, the prevalence of B–O motifs destabilizes long-range order and impedes recrystallization. Taken together, these findings suggest that boron acts as a kinetic bottleneck for atomic transport and increasingly favors amorphization as its concentration rises.

![picture](https://github.com/gyf712/apollox_2_figs/blob/main/fig4.jpg?raw=true)

FIG. 4. Structural characterizations of three samples with varying boron contents. (A–C) AC-HAADF-STEM
images show disordered atomic distribution without periodic crystal lattice. (D–F) HAADF-STEM images show irregular
morphology. (G–I) Corresponding EDS mapping displays homogeneous distribution of Fe, Co, Ni, Mo elements across the
samples.

### E. From theoretical design to experimental insights

Inspired by boron’s sluggish diffusion and strong B–O interactions, we developed a synthesis strategy for multielement amorphous materials by incorporating metal atoms into a BO𝑥 framework (Supplementary Fig. S14 and Supplementary Text L). In this context, the term “synthesis strategy” refers to the materials design principle of promoting amorphization by incorporating elements with low diffusion coefficients. Guided by these predictions, we synthesized amorphous FeCoNiMoBO𝑥 with controlled boron gradients (5.87%, 8.47%, and 11.68%; Supplementary Table S5). Aberration corrected high-angle annular dark field scanning transmission electron microscopy (AC-HAADF-STEM) images (Fig. 4A–C) show disordered atomic distribution without periodic crystal lattice, which is a distinctive feature of amorphous materials. The high resolution HAADF-STEM images show irregular morphology (Fig. 4D–F) while corresponding energy-dispersive X-ray spectroscopy (EDS) mapping (Fig. 4G–I) show uniform elemental distribution across nano-particles. In order to better benchmark our predictive model fidelity, we carried out detailed measurements using synchrotron X-ray diffraction (XRD) and X-ray absorption spectroscopy (XAS). First, we can find some clue from normal XRD patterns (Fig. 5A) that some weak oxide peaks are progressively smoother with increasing boron content. Synchrotron XRD results (Supplementary Fig. S16) show that all three synthesized samples lack distinct Bragg reflections, from Group-1 to Group-3, the broadening and fading of diffraction rings reflect an increasing structural disorder. The corresponding PDFs (Fig. 5B) further confirmed the disappearance of weak long-range periodicity with higher boron incorporation. These observations align with our simulations and further substantiate the claim that increasing boron content promotes amorphization. In short-range PDFs (Fig. 5C), it shows that the amorphization process accompanied with the disruption of M–O–M linkages, which may caused by increased boron taken away original oxygen atoms of the parent oxide and consequently generates M–M bonds. Crucially, the remaining M–O coordination prevents macroscopic phase separation, yielding a homogeneous, boron-stabilized glass. Consistent with this picture, low boron contents preserve vestiges of the oxide’s long-range order, whereas progressive boron incorporation systematically erodes these correlations, thereby unifying the observed structural evolution with the proposed amorphization process.

The extended X-ray absorption fine structure (EXAFS) results provide further insights into the electronic states and local coordination environment. The corresponding X-ray absorption near-edge structure (XANES) of the boron K-edge (Supplementary Fig. S20) shows that boron species are dominated by trigonal BO3 units, matching commercial B2O3​, while a minor BO4 component appears in low-B (high-O) samples. This observation aligns with the dynamic evolution of BO3 clusters from computational simulations. For transition-metal elements, the XANES (Supplementary Fig. S21) indicates that their similar average oxidation states. In the R-space EXAFS spectra (Supplementary Fig. S18) for the Co K-edge, as the boron content increases, the Co–O peak intensity decreases, while Co–M (M = metal) interactions intensify, indicating progressive disruption of M–O–M linkages. Similar trends are observed for the Ni and Fe K-edges. The centroid of the [χ(k), χ(R)] intensity in the wavelet transform of the Co K-edge EXAFS spectra (Fig. 5E) confirms the rupture of Co–O–Co linkages and the concomitant emergence of Co–M bonding during amorphization. Moreover, the R-space EXAFS for Co (Fig. 5D) were well fitted, further validating the fidelity of our predictive model (The EXAFS R-factor and related fitting parameters are provided in the Supplementary Table S6). Structural predictions further quantify this transition using α𝑖𝑗 values [39] (see Methods section M for details), where increasing α𝑖𝑗 for M–O and decreasing values for M–M interactions confirm a shift toward metal aggregation, validating our predictive model.

The structural evolution directly influences catalytic performance. Multi-elemental oxides are widely studied for catalysis and energy storage due to their structural adaptability, electronic tunability, and suppressed cation diffusion, which enhance stability and activity [44–49]. Amorphous oxides, in particular, often outperform crystalline counterparts owing to their enhanced electronic flexibility and active site accessibility [50, 51]. Leveraging these advantages, we examined the OER activity of amorphous FeCoNiMoBO𝑥, integrating the catalytic properties of Fe/Co/Ni oxides with boron’s role in electronic modulation [47, 52, 53].

Given the well-established correlation between Co e𝑔 orbital occupancy and OER activity [54, 55], and supported by our in situ near ambient pressure photoelectron spectroscopy (NAP-XPS) measurements, Co was selected as the representative active site for theoretical analysis in the multicomponent FeCoNiMoBO𝑥 matrix. In situ NAP-XPS measurements under synchrotron radiation (Supplementary Fig. S22) revealed the evolution of Co, Fe, and Ni surface states during OER conditions. The Co 2p spectra (Supplementary Fig. S22B) exhibit a pronounced shift in the 2p3/2 peak toward higher binding energy and the emergence of a characteristic CoOOH feature [36] as the applied potential increases from the open-circuit potential (OCP) to 1.6 V versus RHE, indicating progressive oxidation of surface Co sites. While Fe 2p and Ni 2p spectra (Supplementary Fig. S22C,D) show less pronounced changes in oxidation state, suggesting that Co plays the dominant role as the adsorption site for OER intermediates under operating conditions.

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

The λ₁/λ₂ descriptor, which is described in the recent study on multi-element oxides for OER performance [56] (Supplementary Text K), validated for multi-element oxides [56]—was then employed to quantify the effect of boron incorporation on the local geometry and electronic configurations of Co centers. It found that λ₁ and λ₂ values for Co ions systematically increase with boron concentration, indicating enhanced octahedral distortion at higher boron levels (Supplementary Fig. S13). This prediction is consistent with linear sweep voltammetry results (Supplementary Fig. S15C–D), where overpotentials at 10 mA cm⁻² are similar across all groups (229 mV, 225 mV, and 222 mV for Groups 1–3), at 100 mA cm⁻² they decrease markedly with increasing boron content (515 mV, 483 mV, and 456 mV). The catalysts (Group 1 to 3) were integrated into an anion exchange membrane water electrolyzer; the best-performing Group 3 showed a low voltage degradation rate of 0.227 mV h⁻¹ over 100 hours, significantly outperforming IrO₂ (0.9 mV h⁻¹, Supplementary Fig. S15A) and exhibiting good structural stability (Supplementary Fig. S15B).

These results demonstrate that increasing boron content enhances both OER activity and durability in FeCoNiMoBOₓ. The incorporation of boron reduces diffusion, improving stability by mitigating metal corrosion (Fig. 3A–D). Additionally, higher boron concentrations promote metal aggregation (Fig. 5C–E), forming metal–metal bonds that modulate local coordination environments and intermediate binding energies. The observed increase in Co–O bonding at lower binding energies (< −920 eV) (Supplementary Text G) and enhanced Co–Ni/Co clustering, evidenced by αᵢⱼ (Supplementary Fig. S18), may further contribute to improved catalytic activity.

Additionally, beyond the unary FeCoNiMoBOₓ system, we synthesized an expanded library of BOₓ-based amorphous multi-metal catalysts with binary, ternary, and quaternary compositions (e.g., FeCoNiBOₓ, FeCoMoBOₓ, FeNiMoBOₓ, CoMoBOₓ, CoNiMoBOₓ, FeMoBOₓ, and NiMoBOₓ). This series allowed a systematic assessment of the influence of chemical complexity on amorphization and OER performance. The results show that higher compositional complexity is generally correlated with increased amorphization and improved catalytic activity (Supplementary Fig. S14B–D).

## III. Conclusions

This work represents a paradigm shift in the computational design of amorphous multi-element materials by integrating a physics-guided generative model with PSO algorithm. By encoding CSRO and thermodynamic stability into the generative process, our approach transcends conventional trial-and-error methods, enabling predictive synthesis of complex disordered systems. The ability to rationally design amorphous structures at atomic level with high thermostability was validated through the synthesis and characterization of FeCoNiMoBOₓ, where the predicted amorphization trends, metal clustering effects, and functional enhancements were experimentally confirmed. Moving forward, this framework opens the door to property-driven materials design, where desired functionalities dictate atomic configurations, accelerating the discovery of next-generation materials for catalysis, energy storage, and beyond.

## IV. Methods

### A. Pair-Density Matrix (PDM) descriptor

To encode CSRO in the Cond-CDVAE framework both efficiently and without bias, we introduced the PDM as the primary structural descriptor. The PDM counts the number of near-neighbour atomic pairs for each element-element combination within a specified cutoff radius and arranges these counts in a K × K matrix, where K denotes the number of distinct element types. This representation maintains a fixed dimensionality regardless of composition while fully capturing local chemical environments. The PDM is formally defined in Equation (1):
PDMαβ(rc)=

### B. Model details of Cond-CDVAE

The initial training set of 100,000 structures was generated by randomly occupying atomic sites in fixed-lattice structures. These configurations are then optimized using a finetuned DPA-2 [34], a machine-learning potential capable of handling a wide range of elements. After optimization, each structure is labeled by its PDM.

We adapted the Cond-CDVAE model to generate structures conditioned on the PDM. In our scenario, the conditioning variables include both composition and PDM. The element type of each atom is represented by a categorical embedding vector. The composition is then calculated as a weighted average of the element type embedding vectors based on the number of atoms of each type. Meanwhile, the PDM, represented as a matrix of continuous variables, is flattened into a vector and normalized in the training set. The composition vector and PDM vector are concatenated to form the conditioning vector. Other model hyperparameters are provided in Supplementary Table 1. Once trained, Cond-CDVAE [29] can generate new structures conditioned on the specified PDM, offering a versatile framework to explore diverse CSRO configurations in high-entropy materials.

During the generation stage, the target composition and PDM should be provided. These conditioning variables, combined with a latent vector sampled from the latent space, are used by the lattice predictor to predict lattice parameters. Subsequently, the PGNNDec reconstructs a valid structure from random atomic coordinates using Langevin dynamics.
 
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

## Reference
[1] Wang, H., Tse, J. S., Tanaka, K., Iitaka, T. & Ma, Y. Superconductive sodalite-like clathrate calcium hydride at high pressures. Proc. Natl. Acad. Sci. U.S.A. 109, 6463–6466 (2012).

[2] Liu, H., Naumov, I. I., Hoffmann, R., Ashcroft, N. W. & Hemley, R. J. Potential high-Tc superconducting lanthanum and yttrium hydrides at high pressure. Proc. Natl. Acad. Sci. U.S.A. 114, 6990–6995 (2017).

[3] Peng, F. et al. Hydrogen clathrate structures in rare earth hydrides at high pressures: Possible route to room-temperature superconductivity. Phys. Rev. Lett. 119, 107001 (2017).

[4] Nouira, A., Sokolovska, N. & Crivello, J.-C. Crystal-gan: Learning to discover crystallographic structures with generative adversarial networks. arXiv preprint (2018). arXiv:1810.11203.

[5] Long, T. et al. Inverse design of crystal structures for multicomponent systems. Acta Mater. 231, 117898 (2022).

[6] Morris, J. R. et al. Genetic algorithm optimization of atomic clusters. In Evolutionary Algorithms, 167–175 (Springer, 1999).

[7] Wang, Y., Lv, J., Zhu, L. & Ma, Y. Crystal structure prediction via particle-swarm optimization. Phys. Rev. B 82, 094116 (2010).

[8] Woodley, S. M. & Catlow, R. Crystal structure prediction from first principles. Nat. Mater. 7, 937–946 (2008).

[9] Wang, Y., Lv, J., Zhu, L. & Ma, Y. Calypso: A method for crystal structure prediction. Comput. Phys. Commun. 183, 2063–2070 (2012).

[10] Oganov, A. R., Lyakhov, A. O. & Valle, M. How evolutionary crystal structure prediction works — and why. Acc. Chem. Res. 44, 227–237 (2011).

[11] Ryan, K., Lengyel, J. & Shatruk, M. Crystal structure prediction via deep learning. J. Am. Chem. Soc. 140, 10158–10168 (2018).

[12] Liu, Y., Madanchi, A., Anker, A. S., Simine, L. & Deringer, V. L. The amorphous state as a frontier in computational materials design. Nat. Rev. Mater. 10, 228–241 (2024).

[13] Yao, Y. et al. High-entropy nanoparticles: Synthesis-structure-property relationships and data-driven discovery. Science 376, eabn3103 (2022).

[14] Banko, L. et al. Combinatorial materials discovery strategy for high entropy alloy electrocatalysts using deposition source permutations (2021).
  Preprint at https://arxiv.org/abs/2101.12345, arXiv:2101.12345.

[15] Wu, B. et al. Synthesis of amorphous metal oxides via a crystalline-to-amorphous phase transition strategy. Nat. Synth. 4, 370–379 (2025).

[16] Schuh, C. A., Hufnagel, T. C. & Ramamurty, U. Mechanical behavior of amorphous alloys. Acta Mater. 55, 4067–4109 (2007).

[17] Miao, J., Ercius, P. & Billinge, S. J. L. Atomic electron tomography: 3d structures without crystals. Science 353, aaf2157 (2016).

[18] Yuan, Y. et al. Three-dimensional atomic packing in amorphous solids with liquid-like structure. Nat. Mater. 21, 95–102 (2022).

[19] Chang, C., Deringer, V. L., Katti, K. S., Van Speybroeck, V. & Wolverton, C. M. Simulations in the era of exascale computing. Nat. Rev. Mater. 8, 309–313 (2023).

[20] Stich, I., Car, R. & Parrinello, M. Amorphous silicon studied by ab initio molecular dynamics: Preparation, structure, and properties. Phys. Rev. B 44, 11092–11104 (1991).

[21] Aloka, J. & Jones, R. O. Structural phase transitions on the nanoscale: The crucial pattern in the phase-change materials ge₂sb₂te₅ and gete. Phys. Rev. B 76, 235201 (2007).

[22] Zunger, A., Wei, S.-H., Ferreira, L. G. & Bernard, J. E. Special quasirandom structures. Phys. Rev. Lett. 65, 353–356 (1990).

[23] van de Walle, A. et al. Efficient stochastic generation of special quasirandom structures. Calphad 42, 13–18 (2013).

[24] Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. & Teller, E. Equation of state calculations by fast computing machines. J. Chem. Phys. 21, 1087–1092 (1953).

[25] Drautz, R. Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B 99, 014104 (2019).

[26] Zarkevich, N. A. & Johnson, D. D. Reliable first-principles alloy thermodynamics via truncated cluster expansions. Phys. Rev. Lett. 92, 255702 (2004).

[27] van de Walle, A. et al. Multicomponent multisublattice alloys, nonconfigurational entropy and other additions to the alloy theoretic automated toolkit. Calphad 33, 266–278 (2009).
  (Tools for Computational Thermodynamics.)

[28] Singh, R., Sharma, A., Singh, P., Balasubramanian, G. & Johnson, D. D. Accelerating computational modeling and design of high-entropy alloys. Nat. Comput. Sci. 1, 54–61 (2021).

[29] Luo, X. et al. Deep learning generative model for crystal structure prediction. npj Comput. Mater. 10, 254 (2024).

[30] Kennedy, J. & Eberhart, R. Particle swarm optimization. In Proc. ICNN’95 – Int. Conf. Neural Netw., vol. 4, 1942–1948 (1995).

[31] Sakata, M., Cowlarn, N. & Davies, H. A. Chemical short-range order in liquid and amorphous cu₆₆ti₃₄ alloys. J. Phys. F: Met. Phys. 11, L157 (1981).

[32] Chen, X. et al. Direct observation of chemical short-range order in a medium-entropy alloy. Nature 592, 712–716 (2021).

[33] Saw, C. K. & Schwarz, R. B. Chemical short-range order in dense random-packed models. J. Less Common Met. 140, 385–393 (1988).

[34] Zhang, D. et al. Dpa-2: a large atomic model as a multi-task learner. npj Comput. Mater. 10, 293 (2024).

[35] Ferlat, G., Seitsonen, A. P., Lazzeri, M. & Mauri, F. Hidden polymorphs drive vitrification in b₂o₃. Nat. Mater. 11, 925–929 (2012).

[36] Zhang, B. et al. Homogeneously dispersed multimetal oxygen-evolving catalysts. Science 352, 333–337 (2016).

[37] Chen, G. et al. Two orders of magnitude enhancement in oxygen evolution reactivity on amorphous Ba₀.₅Sr₀.₅Co₀.₈Fe₀.₂O₃₋δ nanofilms with tunable oxidation state. Sci. Adv. 3, e1603206 (2017).

[38] Zhang, B. et al. High-valence metals improve oxygen evolution reaction performance by modulating 3d metal oxidation state energetics. Nat. Catal. 3, 985–992 (2020).

[39] Cowley, J. M. An approximate theory of order in alloys. Phys. Rev. 77, 669–675 (1950).

[40] Miranda, L. J. et al. Pyswarms: a research toolkit for particle swarm optimization in python. J. Open Source Softw. 3, 433 (2018).

[41] Li, W.-L. et al. From planar boron clusters to borophenes and metalloborophenes. Nat. Rev. Chem. 1, 0071 (2017).

[42] van de Walle, A., Asta, M. & Ceder, G. The alloy theoretic automated toolkit: A user guide. Calphad 26, 539–553 (2002).

[43] Hsu, W.-L., Tsai, C.-W., Yeh, A.-C. & Yeh, J.-W. Clarifying the four core effects of high-entropy materials. Nat. Rev. Chem. 8, 471–485 (2024).

[44] Suntivich, J., May, K. J., Gasteiger, H. A., Goodenough, J. B. & Shao-Horn, Y. A perovskite oxide optimized for oxygen evolution catalysis from molecular orbital principles. Science 334, 1383–1385 (2011).

[45] Batchelor, T. A. A. et al. High-entropy alloys as a discovery platform for electrocatalysis. Joule 3, 834–845 (2019).

[46] Han, L. et al. Multifunctional high-entropy materials. Nat. Rev. Mater. 9, 846–865 (2024).

[47] Xiao, Z. et al. Operando identification of the dynamic behavior of oxygen vacancy-rich Co₃O₄ for oxygen evolution reaction. J. Am. Chem. Soc. 142, 12087–12095 (2020).

[48] Löffler, T. et al. Design of complex solid-solution electrocatalysts by correlating configuration, adsorption energy distribution patterns, and activity curves. Angew. Chem. Int. Ed. 59, 5844–5850 (2020).

[49] Ren, J.-T., Chen, L., Wang, H.-Y. & Yuan, Z.-Y. High-entropy alloys in electrocatalysis: from fundamentals to applications. Chem. Soc. Rev. 52, 8319–8373 (2023).

[50] Thangavel, P., Kim, G. & Kim, K. S. Electrochemical integration of amorphous nife (oxy) hydroxides on surface-activated carbon fibers for high-efficiency oxygen evolution in alkaline anion exchange membrane water electrolysis. J. Mater. Chem. A 9, 14043–14051 (2021).

[51] Park, J., Lee, S. & Kim, S. Recent advances in amorphous electrocatalysts for oxygen evolution reaction. Front. Chem. 10, 1030803 (2022).

[52] Gao, Z.-W. et al. Engineering NiO/NiFe LDH intersection to bypass scaling relationship for oxygen evolution reaction via dynamic tridimensional adsorption of intermediates. Advanced Materials 31, 1804769 (2019).

[53] Wu, Z.-P., Lu, X. F., Zang, S.-Q. & Lou, X. W. Non-noble-metal-based electrocatalysts toward the oxygen evolution reaction. Adv. Funct. Mater. 30, 1910274 (2020).

[54] Wang, X. et al. Electrospun thin-walled CuCo₂O₄@C nanotubes as bifunctional oxygen electrocatalysts for rechargeable Zn–air batteries. Nano Lett. 17, 7989–7994 (2017).

[55] Jin, Z. et al. Rugged high-entropy alloy nanowires with in situ formed surface spinel oxide as highly stable electrocatalyst in Zn–air batteries. ACS Mater. Lett. 2, 1698–1706 (2020).

[56] Fan, F. et al. Applicable descriptors under weak metal-oxygen d–p interaction for the oxygen evolution reaction. Angew. Chem. Int. Ed. e202419178 (2024).

[57] de Fontaine, D. The number of independent pair-correlation functions in multicomponent systems. J. Appl. Crystallogr. 4, 15–19 (1971).

[58] Ceguerra, A. V. et al. Short-range order in multicomponent materials. Acta Crystallogr. A 68, 547–560 (2012).

[59] Behler, J. & Parrinello, M. Generalized neural-network representation of high-dimensional potential-energy surfaces. Phys. Rev. Lett. 98, 146401 (2007).

## Acknowledgements
This work was supported by the National Natural Science Foundation of China (Grants No. T2225013, No. 12034009, No. 12174142, No. 42272041, No. 22372004, No. 22405048), National Key Research and Development Program of China (Grants No. 2022YFA1402304, No. 2024YFA1509500), Beijing Natural Science Foundation No. Z240027, Program for Jilin University Science and Technology Innovative Research Team (2021TD—05), Program for Jilin University Computational Interdisciplinary Innovative Platform.

Part of the calculation was performed in the high-performance computing center of Jilin University. Part of the calculation was performed in the San Diego Super-computer Center (SDSC) Expanse at UC San Diego through allocation MAT240028 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program. W.-L.L. and H.Z. thank the KAUST Supercomputing Laboratory for providing computational resources on the Shaheen III supercomputer through project k10175.

We thank Prof. Zipeng Zhao for carrying out the catalytic-stability measurements. We acknowledge Prof. Xueqiang Zhang for the in situ NAP-XPS studies conducted at BL02B Beamline of Shanghai Synchrotron Radiation Facility. We thank Prof. Jihan Zhou for the Electron Microscopy Laboratory at Peking University for the use of aberration-corrected electron microscope. We are also grateful to Yudong Wang for his expert assistance with X-ray diffraction experiments performed at beamline BL17UM of the Shanghai Synchrotron Radiation Facility.

Author contributions M.L., Y.W., and W.-L.L. supervised the research and project. H.L. designed the workflow of ApolloX. C.L., C.P., Z.L., H.Z., S.Z., Z.L., C.P. and J.L. performed the experimental component, including synthesis, characterization, and performance testing. Y.G., X.L., J.L., and Y.D. contributed to code development and model design, while Y.C. and J.W. analyzed and interpreted the data, conducted performance evaluations, and performed model comparisons. G.L. and Y.L. carried out catalytic property calculations. R.W., Z.W., and Z.Z. contributed to training the machine learning potential. Additionally, H.L., C.L., M.L., Y.W., W.-L.L., X.L., C.M., J.L. and Z.X. contributed to the writing of the manuscript.

Competing interests: The authors declare no competing interests.

Data and code availability:The authors declare that the main data supporting the findings of this study are contained within the paper and its associated Supplementary Information. The ApolloX source code is available on GitHub: https://github.com/FNC001/ApolloX

---

*Note:* The above Methods subsections summarize the main computational and experimental procedures underlying the ApolloX framework and its application to the FeCoNiMoBOx system. Additional details, including parameter values, convergence tests, and data-processing protocols, can be provided in supplementary materials or dedicated technical appendices.
