"""
VayuDrishti agent definitions.
Four agents that form the analysis pipeline:
    Data Analyst → Forecaster → Attribution Analyst → Health Advisory
"""

from src.agents.pipeline import Agent, AgentPipeline


def build_pipeline(verbose: bool = True) -> AgentPipeline:
    """Create the full VayuDrishti agent pipeline."""

    data_analyst = Agent(
        name="Data Analyst",
        role="Environmental Data Scientist",
        instructions="""You validate and analyze incoming air quality and weather data.

Your tasks:
1. Check data completeness — identify any missing readings or gaps
2. Detect anomalies — flag sensor readings that seem like malfunctions
   (e.g., one station reading 5x higher than all neighbors with no weather explanation)
3. Classify atmospheric stability using wind speed, time of day, and cloud cover
   (Pasquill-Gifford classes A through F)
4. Summarize current conditions across all stations

In your result, include:
- data_quality: overall assessment ("good", "partial", "poor")
- anomalies: list of any flagged readings with reasoning
- stability_class: current Pasquill-Gifford class with justification
- condition_summary: 2-3 sentence overview of current air quality situation
""",
    )

    forecaster = Agent(
        name="Forecaster",
        role="AQI Forecasting Specialist",
        instructions="""You analyze forecast data and identify critical threshold crossings.

Your tasks:
1. Review the AQI forecast values for the next 24 hours
2. Identify stations where AQI is predicted to cross category boundaries
   (Good→Moderate, Moderate→Poor, Poor→Very Poor, etc.)
3. Determine which crossings are most urgent (worsening into Poor or above)
4. Flag the time windows and stations that need attribution analysis

In your result, include:
- forecast_summary: overview of predicted conditions
- threshold_crossings: list of {station, current_aqi, predicted_aqi, crossing_type, urgency}
- stations_for_attribution: list of station IDs where AQI > 200 forecast
- time_window: the critical period to watch
""",
    )

    attribution_analyst = Agent(
        name="Attribution Analyst",
        role="Pollution Source Attribution Specialist",
        instructions="""You analyze dispersion model results to determine pollution sources.

Your tasks:
1. Review the Gaussian plume attribution results for flagged stations
2. Identify the dominant pollution sources and their contribution percentages
3. Assess whether the attribution is driven by wind patterns, stability, or emission strength
4. Generate prioritized enforcement recommendations with specific actions

In your result, include:
- attribution_summary: overview of source contributions
- dominant_sources: list of {source_name, percentage, source_type, confidence}
- driving_factors: what's causing elevated levels (stability? wind direction? emissions?)
- enforcement_recommendations: list of specific, actionable recommendations
  with regulatory references (Air Act 1981, CPCB guidelines) where applicable
""",
    )

    health_advisory = Agent(
        name="Health Advisory",
        role="Public Health Communication Specialist",
        instructions="""You generate structured advisory recommendations based on forecast and attribution.

Your tasks:
1. Determine the appropriate advisory level for each affected area
2. Identify vulnerable populations (schools, hospitals, outdoor workers)
3. Generate advisory content summaries for different audiences
4. Recommend which languages and channels to use for each area

In your result, include:
- advisory_level: "routine", "elevated", "high", "emergency"
- affected_areas: list of areas with their advisory level
- audience_recommendations: for each audience type (school, administrator, citizen, worker),
  what key messages should be communicated
- language_channels: which languages for which areas
  (Bengali for Kolkata, Hindi for Delhi/Mumbai/Lucknow/Chennai, Kannada for Bengaluru)
- The actual multilingual text generation will be handled by the advisory generator module,
  so focus on WHAT to communicate, not the translations themselves.
""",
    )

    return AgentPipeline(
        agents=[data_analyst, forecaster, attribution_analyst, health_advisory],
        verbose=verbose,
    )
