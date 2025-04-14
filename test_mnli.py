import os
import torch.nn.functional as F
import pandas as pd
from apsr_mnli_analysis import MNLIAnalysis
from apsr_utils import APSRUtils

# Define thresholds for high, medium, and low scores.
THRESHOLDS = {
    "high": 0.7,  # Scores >= 0.7 are considered high.
    "low": 0.3,  # Scores <= 0.3 are considered low.
}

# Define test cases with increasing complexity.
test_cases = [
    # 1–12: Original simple cases (low complexity).
    {
        "test_abstract_a": "The sky is blue.",
        "test_abstract_b": "The sky is not blue.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "Water is wet.",
        "test_abstract_b": "Water is wet.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "Cats are mammals.",
        "test_abstract_b": "Cats are animals.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "The Earth revolves around the Sun.",
        "test_abstract_b": "The Sun revolves around the Earth.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "Birds can fly.",
        "test_abstract_b": "Penguins cannot fly.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "Dogs are loyal animals.",
        "test_abstract_b": "Dogs are often considered loyal.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "Climate change is caused by human activities.",
        "test_abstract_b": "Human activities are the primary driver of climate change.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "Electric cars are more environmentally friendly than gas cars.",
        "test_abstract_b": "Gas cars are more environmentally friendly than electric cars.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "The study found that exercise improves mental health.",
        "test_abstract_b": "Exercise has no impact on mental health.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "The research highlights the importance of biodiversity in ecosystems.",
        "test_abstract_b": "Biodiversity plays a critical role in maintaining ecosystem stability.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "The experiment demonstrated that high temperatures reduce crop yields.",
        "test_abstract_b": "High temperatures have no significant effect on crop yields.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "Vaccines are effective in preventing diseases.",
        "test_abstract_b": "Vaccines are ineffective in preventing diseases.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "low_complexity_test",
    },
    {
        "test_abstract_a": "The economy is growing.",
        "test_abstract_b": "The economic growth is robust.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Economic indicators show a downturn.",
        "test_abstract_b": "The market is performing well.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Consumer confidence has increased.",
        "test_abstract_b": "The sentiment among consumers has risen.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The central bank raised interest rates.",
        "test_abstract_b": "Interest rates remained unchanged by the central bank.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Unemployment has decreased in the last quarter.",
        "test_abstract_b": "The jobless rate dropped over the previous quarter.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Government spending on infrastructure has grown.",
        "test_abstract_b": "There was a reduction in infrastructure investment by the government.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The fiscal deficit is narrowing.",
        "test_abstract_b": "The government is experiencing a smaller budget gap.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The trade surplus widened this year.",
        "test_abstract_b": "The country's exports exceeded imports more significantly.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Political instability disrupted economic activities.",
        "test_abstract_b": "Economic performance was unaffected by political unrest.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Voter participation remained consistent over the past elections.",
        "test_abstract_b": "The electoral turnout showed minimal variation between elections.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    # 23–32: Cases with more nuanced political topics (medium complexity).
    {
        "test_abstract_a": "Public trust in government institutions has eroded.",
        "test_abstract_b": "Citizens remain indifferent towards government institutions.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Regulatory reforms have improved market efficiency.",
        "test_abstract_b": "New regulations have optimized market performance.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The diplomatic negotiations resulted in significant policy shifts.",
        "test_abstract_b": "International discussions led to notable changes in policy.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The study finds no connection between trade policies and domestic job growth.",
        "test_abstract_b": "The analysis confirms that trade policies are the main driver of domestic job creation.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The economic model predicts a moderate rise in GDP.",
        "test_abstract_b": "Forecasts indicate a considerable expansion in the gross domestic product.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Recent legislative debates suggest a shift towards progressive policies.",
        "test_abstract_b": "Parliamentary discussions have increasingly supported progressive reforms.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The study observed a decline in public sector employment.",
        "test_abstract_b": "Public sector jobs increased according to the report.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Economic liberalization policies have reduced market distortions.",
        "test_abstract_b": "Market interventions continue to cause imbalances in the economy.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The policy analysis highlights the benefits of decentralization.",
        "test_abstract_b": "Centralization remains the preferred approach, as per the study.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "There is substantial evidence that immigration contributes to economic growth.",
        "test_abstract_b": "Immigration has no significant impact on the economy, according to recent research.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Fiscal austerity measures were implemented to stabilize the economy.",
        "test_abstract_b": "The government adopted austerity measures aiming to restore economic balance.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The outcome of the referendum remained ambiguous.",
        "test_abstract_b": "Results from the vote were inconclusive.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The research did not find any significant correlation between political ideology and tax compliance.",
        "test_abstract_b": "The study shows no meaningful link between political views and adherence to tax policies.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Socioeconomic disparities are widening in urban areas.",
        "test_abstract_b": "The report indicates an increase in urban inequality.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The study undermines the theoretical framework of voter behavior.",
        "test_abstract_b": "The analysis supports the conventional theory of voter decision-making.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Bureaucratic reforms have yielded positive outcomes in administrative efficiency.",
        "test_abstract_b": "The reforms in public administration have not improved efficiency.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Political alliances are often fluid and subject to change.",
        "test_abstract_b": "The study emphasizes the transient nature of political coalitions.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Foreign aid has bolstered economic development in recipient countries.",
        "test_abstract_b": "The analysis reveals that international assistance has failed to stimulate growth.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "The policy clearly outlines measures for economic recovery.",
        "test_abstract_b": "The plan lacks detailed strategies for reviving the economy.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    {
        "test_abstract_a": "Analytical models suggest a correlation between social media usage and political mobilization.",
        "test_abstract_b": "Empirical research demonstrates that online engagement does not predict political activism.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "medium_complexity_test",
    },
    # 43–50: Complex cases with a political science flavor (high complexity).
    {
        "test_abstract_a": "The cross-national study presents consistent findings regarding voter turnout.",
        "test_abstract_b": "Data from multiple countries indicates stable levels of electoral participation.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "high_complexity_test",
    },
    {
        "test_abstract_a": "Increased globalization is linked to cultural homogenization.",
        "test_abstract_b": "The global integration process shows varying degrees of cultural diversity retention.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "high_complexity_test",
    },
    {
        "test_abstract_a": "The policy intervention had no observable effect on reducing corruption.",
        "test_abstract_b": "The government’s anti-corruption measures did not yield measurable results.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "high_complexity_test",
    },
    {
        "test_abstract_a": "The analysis reveals that media framing influences public perception significantly.",
        "test_abstract_b": "The study found only a marginal impact of media representation on public opinion.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "high_complexity_test",
    },
    {
        "test_abstract_a": "The research underscores the complexity of electoral behavior.",
        "test_abstract_b": "The study illustrates that voter decision-making is overly simplified.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "high_complexity_test",
    },
    {
        "test_abstract_a": "The longitudinal study offers comprehensive insights into policy evolution.",
        "test_abstract_b": "The research captures the dynamic transformation of political strategies over time.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "high_complexity_test",
    },
    {
        "test_abstract_a": "Recent reforms have catalyzed transformative changes in governance.",
        "test_abstract_b": "The study suggests that recent policy updates resulted in only modest governance improvements.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "high_complexity_test",
    },
    {
        "test_abstract_a": "This paper examines the intersection of technology, policy, and society in the context of modern democracies.",
        "test_abstract_b": "The research investigates how technological advancements influence governmental policies and societal trends within contemporary democratic frameworks.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "high_complexity_test",
    },
    {
        "test_abstract_a": "This study employs a mixed-methods approach to examine the impact of institutional reforms on democratic consolidation in post-colonial states, analyzing a longitudinal dataset spanning multiple geopolitical regions with an emphasis on legislative efficiency and administrative accountability.",
        "test_abstract_b": "The research uses a mixed-methods design to explore how institutional reforms affect democratic consolidation in post-colonial states, employing longitudinal data and focusing on legislative efficiency.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "Our comprehensive analysis of electoral volatility across emerging democracies indicates that decentralization measures have significantly mitigated central authoritarian tendencies, fostering more robust local governance over time.",
        "test_abstract_b": "The study contends that decentralization efforts in emerging democracies have exacerbated authoritarian control, thereby undermining local governance structures and diminishing participatory politics.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "The paper scrutinizes the interplay between globalization and national identity formation, employing structural equation modeling to assess the mediating role of media influence over a fifty-year period.",
        "test_abstract_b": "This study investigates how globalization influences national identity formation using quantitative models over several decades, revealing effects that are contingent on multiple intervening variables.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "By integrating game theory with historical institutionalism, this research delineates the causal mechanisms through which political incumbency shapes policy stability, drawing on case studies from advanced democracies in the late twentieth century.",
        "test_abstract_b": "This research integrates game theory with historical institutionalism to explain the impact of political incumbency on policy stability, supported by case studies from late twentieth-century advanced democracies.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "The empirical findings reveal that increasing fiscal decentralization correlates positively with regional economic disparities, thereby challenging classical theories that advocate for a homogenizing effect of state structures.",
        "test_abstract_b": "The analysis clearly demonstrates that fiscal decentralization leads to uniform economic growth and reduces regional disparities, reinforcing traditional expectations of centralized control.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "This article explores the dialectical relationship between public opinion and legislative responsiveness, leveraging extensive survey data to map trends over multiple election cycles while highlighting both alignment and divergence in the observed dynamics.",
        "test_abstract_b": "The study examines the relationship between public opinion and legislative action, using survey data across several election cycles, and finds a balance of convergent and divergent trends in policy responsiveness.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "Drawing upon network theory, the paper explicates how transnational policy diffusion accelerates reform adoption in domestic political institutions, with empirical tests substantiating a significant positive correlation.",
        "test_abstract_b": "Utilizing network theory, this study demonstrates that transnational policy diffusion expedites the adoption of reforms within domestic political institutions, supported by empirical evidence of a positive correlation.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "The study asserts that the decentralization of fiscal authority resulted in increased accountability and improved citizen satisfaction in administrative governance.",
        "test_abstract_b": "The research indicates that decentralizing fiscal power led to diminished accountability and lower levels of citizen satisfaction in administrative governance.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "A critical examination of ideological polarization reveals complex interactions among media ecosystems, political narratives, and voter behavior, wherein the empirical data does not decisively support a unidirectional causality.",
        "test_abstract_b": "The research scrutinizes the nexus between media, political narratives, and voter behavior, finding that the data reflects multifaceted interactions rather than a clear-cut cause-and-effect relationship.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "In a multivariate framework, this paper investigates the symbiotic effects of international trade agreements and domestic policy shifts on economic liberalization, offering robust evidence for a reinforcing cycle between them.",
        "test_abstract_b": "This study employs a multivariate approach to illustrate how international trade agreements synergize with domestic policy shifts, providing strong evidence of a reinforcing cycle that promotes economic liberalization.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "The longitudinal analysis suggests that the introduction of term limits has fundamentally reshaped political accountability, thereby enhancing democratic practices across a variety of regimes.",
        "test_abstract_b": "This investigation finds that term limits have had a negligible impact on political accountability, failing to significantly alter democratic practices in most regimes.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "Focusing on the role of institutional legacies, the article assesses the extent to which historical socioeconomic stratification informs contemporary policy outcomes through a series of regression discontinuity designs.",
        "test_abstract_b": "The paper evaluates the influence of historical socioeconomic legacies on modern policy outcomes using regression techniques, revealing an effect that is moderate and context-dependent.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "By employing a comparative case study methodology, the investigation delineates how divergent welfare regimes influence the distribution of social benefits, ultimately fostering enhanced social equity.",
        "test_abstract_b": "This comparative study shows that differing welfare regimes significantly affect the distribution of social benefits, thereby promoting greater social equity.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "The quantitative analysis uncovers that political decentralization invariably strengthens local governance by increasing participatory avenues and reducing bureaucratic inertia.",
        "test_abstract_b": "Contrary to prevailing theories, the statistical findings suggest that political decentralization does not consistently enhance local governance, as participatory levels and bureaucratic efficiency remain largely unchanged.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "This study employs a mixed-methods design to explore the interplay between political campaign financing and legislative behavior, suggesting that while funding sources may influence policy preferences, the effect is contingent on multiple contextual factors.",
        "test_abstract_b": "Using both qualitative and quantitative approaches, the research investigates how campaign financing shapes legislative decisions, yielding findings that indicate a conditional relationship moderated by several political contexts.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "Leveraging a robust dataset from international organizations, the paper argues that institutional reforms aimed at enhancing transparency lead to measurable improvements in government performance, thereby reinforcing democratic accountability.",
        "test_abstract_b": "This empirical study, based on extensive data from global agencies, provides compelling evidence that transparency-driven institutional reforms significantly boost government performance and democratic accountability.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "The research posits that electoral system reforms have catalyzed a substantive shift towards increased political pluralism and competitive governance.",
        "test_abstract_b": "Analyses indicate that electoral system reforms have had limited impact, failing to significantly alter entrenched patterns of political monopolization.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "This extensive study explores the mediating effects of globalization on state autonomy, employing a diverse set of case studies to interrogate the balance between national sovereignty and international pressures.",
        "test_abstract_b": "The study examines how globalization interacts with state autonomy across multiple cases, yielding outcomes that do not uniformly favor either enhanced sovereignty or increased external influence.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "In an interdisciplinary approach, the paper integrates sociological theory with political analysis to demonstrate that increasing civic engagement is a direct outcome of policy reforms aimed at governmental transparency.",
        "test_abstract_b": "This interdisciplinary research shows that policy reforms focused on transparency lead to higher levels of civic engagement, as evidenced by both sociological and political data.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "The comprehensive model posits that upward social mobility is substantially driven by educational reforms within public institutions, thereby challenging traditional notions of class rigidity.",
        "test_abstract_b": "Findings reveal that educational reforms have minimal impact on upward social mobility, affirming long-held theories regarding entrenched class structures.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "Employing a combination of factor analysis and case studies, the paper examines the extent to which policy innovations in environmental governance are shaped by both domestic political pressures and international normative frameworks.",
        "test_abstract_b": "The research investigates environmental governance reforms using both quantitative and qualitative lenses, concluding that the influences of domestic politics and international norms are balanced and ambiguous.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "The article systematically deconstructs the relationship between legislative oversight and executive accountability, revealing that stringent oversight mechanisms contribute significantly to institutional checks and balances.",
        "test_abstract_b": "This study demonstrates that robust legislative oversight directly enhances executive accountability, thereby reinforcing the system of institutional checks and balances.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "Analyzing over three decades of political data, the study concludes that decentralization policies have led to a measurable erosion of central authority, thereby facilitating a more dispersed model of governance.",
        "test_abstract_b": "Contrary to expectations, the longitudinal study finds that decentralization policies did not weaken central authority and, in some cases, consolidated it further.",
        "expected": {
            "contradiction_prob": "high",
            "neutral_prob": "low",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "This investigation synthesizes multiple analytical frameworks to explore the causal linkages between party ideology, economic policy shifts, and societal outcomes, suggesting that the interdependencies are complex and context-specific.",
        "test_abstract_b": "Integrating diverse theoretical perspectives, the study assesses how party ideology interacts with economic policies and their societal impacts, yielding nuanced and indeterminate results.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "high",
            "entailment_prob": "low",
        },
        "complexity_level": "very_high_complexity_test",
    },
    {
        "test_abstract_a": "Integrating advanced econometric techniques with qualitative narrative analysis, the paper robustly demonstrates that sustained political reforms produce enduring improvements in governance quality and heightened citizen trust across varied political systems.",
        "test_abstract_b": "This comprehensive analysis, which combines econometric modeling with narrative inquiry, provides strong evidence that persistent political reforms lead to lasting enhancements in governance quality and increased citizen trust.",
        "expected": {
            "contradiction_prob": "low",
            "neutral_prob": "low",
            "entailment_prob": "high",
        },
        "complexity_level": "very_high_complexity_test",
    },
]


def AssignPassFail(expected: dict, actual: dict) -> str:
    result = "strong pass"

    for key, expected_value in expected.items():
        actual_value = actual[key]

        if expected_value == "high" and actual_value >= THRESHOLDS["high"]:
            continue
        elif expected_value == "low" and actual_value <= THRESHOLDS["low"]:
            continue
        elif (
            expected_value == "medium"
            and THRESHOLDS["low"] < actual_value < THRESHOLDS["high"]
        ):
            continue
        else:
            # If any condition fails, downgrade the result.
            result = "fail" if result == "strong pass" else result
            result = "pass" if result == "fail" else result

    return result


# Helper function to assign thresholds to probabilities.
def AssignThreshold(probability: float) -> str:
    if probability >= THRESHOLDS["high"]:
        return "high"
    elif probability <= THRESHOLDS["low"]:
        return "low"
    else:
        return "medium"


# Test the SubmitMNLIWork function with predefined test cases.
def TestSubmitMNLIWork(
    output_csv_path: str = "./output_data/test_mnli_output.csv",
) -> None:
    utils = APSRUtils(log_path="./logs/test_mnli_log.txt")

    print("Initializing AbstractAnalysis class...")
    ab_an = MNLIAnalysis(load_existing_cache=False)

    # Process each test case.
    print("Running test cases...")
    results = []
    total_cases = len(test_cases)

    for idx, case in enumerate(test_cases, start=1):
        test_abstract_a = case["test_abstract_a"]
        test_abstract_b = case["test_abstract_b"]
        expected = case["expected"]
        complexity_level = case["complexity_level"]

        try:
            # Call SubmitMNLIWork to compute actual probabilities.
            probabilities: F.Tensor = ab_an.SubmitMNLIWork(
                test_abstract_a, [test_abstract_b]
            )
            actual = {
                "contradiction_prob": round(probabilities[0, 0].item(), 4),
                "neutral_prob": round(probabilities[0, 1].item(), 4),
                "entailment_prob": round(probabilities[0, 2].item(), 4),
            }

            # Assign thresholds to probabilities.
            thresholds = {
                "determined_contradiction_threshold": AssignThreshold(
                    actual["contradiction_prob"]
                ),
                "determined_neutral_threshold": AssignThreshold(actual["neutral_prob"]),
                "determined_entailment_threshold": AssignThreshold(
                    actual["entailment_prob"]
                ),
            }

            # Append results.
            results.append(
                {
                    "complexity_level": complexity_level,
                    "test_abstract_a": test_abstract_a,
                    "test_abstract_b": test_abstract_b,
                    #
                    # Contradiction measurements.
                    "expected_contradiction": expected["contradiction_prob"],
                    "determined_contradiction_threshold": thresholds[
                        "determined_contradiction_threshold"
                    ],
                    "actual_contradiction": actual["contradiction_prob"],
                    #
                    # Neutral measurements.
                    "expected_neutral": expected["neutral_prob"],
                    "determined_neutral_threshold": thresholds[
                        "determined_neutral_threshold"
                    ],
                    "actual_neutral": actual["neutral_prob"],
                    #
                    # Entailment measurements.
                    "expected_entailment": expected["entailment_prob"],
                    "determined_entailment_threshold": thresholds[
                        "determined_entailment_threshold"
                    ],
                    "actual_entailment": actual["entailment_prob"],
                }
            )

        except Exception as e:
            # Log any errors during processing.
            fn, line = utils.GetFuncLine()
            utils.Log(
                "error",
                *utils.GetFuncLine(),
                f"Error processing test case {idx}: {e}",
            )
            raise e

        # Update progress bar.
        utils.ProgressBar(idx, total_cases, prefix="Processing Test Cases")

    # Save results to a CSV file.
    utils.OutputCSV(results, output_csv_path, print_notification=True)
    utils.WriteLog()
    print(f"Test results saved to {os.path.abspath(output_csv_path)}")


if __name__ == "__main__":
    TestSubmitMNLIWork()
