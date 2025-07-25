# Proportion of weights to assign to each metric.
FEATURE_WEIGHTS = {
    "compute_units": 0.53,  # Total amount of compute time (compute multiplier * total time).
    "invocation_count": 0.20,  # Total number of invocations.
    "unique_chute_count": 0.20,  # Average instantaneous unique chutes (gpu scaled) over the scoring period.
    "bounty_count": 0.07,  # Number of bounties received (not bounty values, just counts).
}
# Time slice to calculate the incentives from.
SCORING_INTERVAL = "7 days"
# Query to fetch raw metrics for compute_units, invocation_count, and bounty_count.
NORMALIZED_COMPUTE_QUERY = """
SELECT
    mn.hotkey,
    COUNT(*) as invocation_count,
    COUNT(CASE WHEN i.bounty > 0 THEN 1 END) AS bounty_count,
    sum(
        i.bounty +
        i.compute_multiplier *
        CASE
            -- For step-based computations
            WHEN i.metrics->>'steps' IS NOT NULL
                AND (i.metrics->>'steps')::float > 0
                AND i.metrics->>'masps' IS NOT NULL
            THEN (i.metrics->>'steps')::float * (i.metrics->>'masps')::float

            -- For token-based computations (it + ot)
            WHEN i.metrics->>'it' IS NOT NULL
                AND i.metrics->>'ot' IS NOT NULL
                AND (i.metrics->>'it')::float > 0
                AND (i.metrics->>'ot')::float > 0
                AND i.metrics->>'maspt' IS NOT NULL
            THEN ((i.metrics->>'it')::float + (i.metrics->>'ot')::float) * (i.metrics->>'maspt')::float

            -- Fallback to actual elapsed time
            ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
        END
    ) AS compute_units
FROM invocations i
JOIN metagraph_nodes mn ON i.miner_hotkey = mn.hotkey AND mn.netuid = 64
WHERE i.started_at > NOW() - INTERVAL '{interval}'
AND i.error_message IS NULL
AND i.miner_uid >= 0
AND i.completed_at IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM reports
    WHERE invocation_id = i.parent_invocation_id
    AND confirmed_at IS NOT NULL
)
GROUP BY mn.hotkey
ORDER BY compute_units DESC;
"""
# Query to calculate the average number of unique chutes active at any single point in time, i.e. unique_chute_count.
UNIQUE_CHUTE_AVERAGE_QUERY = """
WITH time_series AS (
  SELECT
    generate_series(
      date_trunc('hour', now() - INTERVAL '{interval}'),
      date_trunc('hour', now()),
      INTERVAL '1 hour'
    ) AS time_point
),
-- Get all instances that had at least one successful invocation (ever) while the instance was alive.
instances_with_success AS (
  SELECT DISTINCT
    instance_id
  FROM invocations ii
  WHERE
    error_message IS NULL
    AND completed_at IS NOT NULL
    AND miner_uid >= 0
    AND NOT EXISTS (
        SELECT 1
        FROM reports
        WHERE invocation_id = ii.parent_invocation_id
        AND confirmed_at IS NOT NULL
    )
),
-- Get all unique miner_hotkeys from instance_audit relevant to the network
all_miners AS (
  SELECT DISTINCT ia.miner_uid, ia.miner_hotkey
  FROM instance_audit ia
  JOIN metagraph_nodes mn ON ia.miner_hotkey = mn.hotkey
  WHERE mn.netuid = 64 AND mn.node_id >= 0
),
-- For each time point, find active instances that have had successful invocations
active_instances_per_timepoint AS (
  SELECT
    ts.time_point,
    ia.instance_id,
    ia.chute_id,
    ia.miner_hotkey
  FROM time_series ts
  JOIN instance_audit ia ON
    ia.verified_at <= ts.time_point AND
    (ia.deleted_at IS NULL OR ia.deleted_at >= ts.time_point)
  JOIN metagraph_nodes mn ON ia.miner_hotkey = mn.hotkey
  JOIN instances_with_success iws ON
    ia.instance_id = iws.instance_id
  WHERE mn.netuid = 64 AND mn.node_id >= 0
),
-- Pre-calculate the most recent GPU count for each chute from chute_history
chute_latest_gpu_history AS (
  SELECT DISTINCT ON (ch.chute_id) -- Get only the latest record per chute_id
      ch.chute_id,
      (ch.node_selector->>'gpu_count')::integer AS latest_gpu_count
  FROM chute_history ch
  WHERE ch.node_selector ? 'gpu_count'
    AND jsonb_typeof(ch.node_selector->'gpu_count') = 'number'
  ORDER BY ch.chute_id, ch.created_at DESC
),
-- Calculate GPU-weighted chute count per miner per time point using historical max GPU count
active_chutes_per_timepoint AS (
  SELECT
    aipt.time_point,
    aipt.miner_hotkey,
    -- Sum the latest_gpu_count, defaulting to 1 if no valid history exists for the chute
    SUM(COALESCE(clgh.latest_gpu_count, 1)) AS gpu_weighted_chutes
  FROM (
    -- Get distinct chute_ids per time point and miner
    SELECT DISTINCT
      time_point,
      miner_hotkey,
      chute_id
    FROM active_instances_per_timepoint
  ) aipt
  -- LEFT JOIN with the pre-calculated *latest* GPU count per chute from history
  LEFT JOIN chute_latest_gpu_history clgh ON -- Join with the new CTE
    aipt.chute_id = clgh.chute_id
  GROUP BY aipt.time_point, aipt.miner_hotkey
),
-- Create a cross join of all time points with all miners
all_timepoints_for_all_miners AS (
  SELECT
    ts.time_point,
    am.miner_hotkey
  FROM time_series ts
  CROSS JOIN all_miners am
),
-- Join with active_chutes to get complete dataset with zeros
complete_dataset AS (
  SELECT
    atm.miner_hotkey,
    atm.time_point,
    COALESCE(acpt.gpu_weighted_chutes, 0) AS gpu_weighted_chutes
  FROM all_timepoints_for_all_miners atm
  LEFT JOIN active_chutes_per_timepoint acpt ON
    atm.time_point = acpt.time_point AND
    atm.miner_hotkey = acpt.miner_hotkey
)
-- Calculate average GPU-weighted chutes per miner across all time points
SELECT miner_hotkey, AVG(gpu_weighted_chutes)::integer AS avg_gpu_weighted_chutes
FROM complete_dataset
GROUP BY miner_hotkey
ORDER BY avg_gpu_weighted_chutes DESC;
"""

# Jobs.
JOBS_QUERY = """
WITH

-- Count of miner-terminated jobs in the past 7 days
miner_terminated_counts AS (
    SELECT
        miner_hotkey,
        COUNT(*) as terminated_job_count
    FROM jobs
    WHERE (started_at >= now() - interval '{interval}' OR finished_at >= now() - interval '{interval}')
      AND miner_terminated = true
      AND miner_hotkey IS NOT NULL
    GROUP BY miner_hotkey
),

-- Compute units/counts for currently in-progress jobs.
running_jobs_cus AS (
    SELECT
        miner_hotkey,
        SUM(extract(epoch from (now() - started_at)) * compute_multiplier) as running_cus,
        COUNT(*) as running_job_count
    FROM jobs
    WHERE started_at IS NOT NULL
      AND finished_at IS NULL
      AND miner_hotkey IS NOT NULL
      AND EXISTS (SELECT 1 FROM instances WHERE instance_id = jobs.instance_id)
    GROUP BY miner_hotkey
),

-- Compute units/counts for jobs completed within the interval.
completed_jobs_cus AS (
    SELECT
        miner_hotkey,
        SUM(extract(epoch from (finished_at - started_at)) * compute_multiplier) as completed_cus,
        COUNT(*) as completed_job_count
    FROM jobs
    WHERE finished_at >= now() - interval '{interval}'
      AND miner_terminated = false
      AND started_at IS NOT NULL
      AND miner_hotkey IS NOT NULL
    GROUP BY miner_hotkey
),

-- Combine the results aggregated by hotkeys.
all_miners AS (
    SELECT miner_hotkey FROM miner_terminated_counts
    UNION
    SELECT miner_hotkey FROM running_jobs_cus
    UNION
    SELECT miner_hotkey FROM completed_jobs_cus
)
SELECT
    am.miner_hotkey,
    COALESCE(mt.terminated_job_count, 0) as terminated_jobs,
    COALESCE(rj.running_job_count, 0) as running_jobs,
    COALESCE(cj.completed_job_count, 0) as completed_jobs,
    COALESCE(rj.running_job_count, 0) + COALESCE(cj.completed_job_count, 0) as total_jobs,
    COALESCE(rj.running_cus, 0) as current_running_cus,
    COALESCE(cj.completed_cus, 0) as completed_cus,
    COALESCE(rj.running_cus, 0) + COALESCE(cj.completed_cus, 0) as total_cus
FROM all_miners am
LEFT JOIN miner_terminated_counts mt ON am.miner_hotkey = mt.miner_hotkey
LEFT JOIN running_jobs_cus rj ON am.miner_hotkey = rj.miner_hotkey
LEFT JOIN completed_jobs_cus cj ON am.miner_hotkey = cj.miner_hotkey
ORDER BY terminated_jobs DESC, total_cus DESC;
"""

# Unique chute history.
UNIQUE_CHUTE_HISTORY_QUERY = (
    UNIQUE_CHUTE_AVERAGE_QUERY.replace(
        "SELECT miner_hotkey, AVG", "SELECT miner_hotkey, time_point::text, AVG"
    )
    .replace("GROUP BY miner_hotkey", "GROUP BY miner_hotkey, time_point")
    .replace("ORDER BY avg_gpu_weighted_chutes DESC", "ORDER BY miner_hotkey ASC, time_point DESC")
    .replace(
        "FROM complete_dataset",
        "FROM complete_dataset WHERE miner_hotkey IN (SELECT hotkey FROM metagraph_nodes WHERE netuid = 64)",
    )
)

# Utilization ratio for busiest chutes.
UTILIZATION_THRESHOLD = 0.02
UTILIZATION_RATIO_QUERY = """
WITH instance_spans AS (
  SELECT
    miner_hotkey, instance_id,
    MAX(completed_at) - MIN(started_at) as total_active_time,
    SUM(completed_at - started_at) AS total_processing_time
  FROM invocations
  WHERE started_at >= now() - INTERVAL '{interval}'
  AND error_message IS NULL AND completed_at IS NOT NULL
  GROUP BY miner_hotkey, instance_id
),
instance_metrics AS (
  SELECT
    miner_hotkey, instance_id,
    EXTRACT(EPOCH FROM total_active_time) AS total_active_seconds,
    EXTRACT(EPOCH FROM total_processing_time) AS total_processing_seconds,
    CASE
      WHEN EXTRACT(EPOCH FROM total_active_time) > 0
      THEN ROUND(
        (EXTRACT(EPOCH FROM total_processing_time) /
         EXTRACT(EPOCH FROM total_active_time))::numeric,
        2
      )
      ELSE 0
    END AS busy_ratio
  FROM instance_spans
  JOIN metagraph_nodes mn ON instance_spans.miner_hotkey = mn.hotkey
),
ranked_instances AS (
  SELECT
    miner_hotkey, instance_id,
    total_active_seconds, total_processing_seconds, busy_ratio,
    ROW_NUMBER() OVER (PARTITION BY miner_hotkey ORDER BY busy_ratio DESC) AS rank
  FROM instance_metrics WHERE total_active_seconds >= 3600
),
top_instances AS (
  SELECT
    miner_hotkey, instance_id,
    total_active_seconds, total_processing_seconds, busy_ratio
  FROM ranked_instances
  WHERE rank <= 3
),
instance_counts AS (
  SELECT
    miner_hotkey,
    COUNT(*) AS instance_count
  FROM top_instances
  GROUP BY miner_hotkey
)
SELECT
  mn.hotkey AS miner_hotkey,
  CASE
    WHEN ic.instance_count >= 3 THEN ROUND(MIN(ti.busy_ratio)::numeric, 6)
    ELSE 0
  END AS min_top_busy_ratio
FROM metagraph_nodes mn
LEFT JOIN top_instances ti ON mn.hotkey = ti.miner_hotkey
LEFT JOIN instance_counts ic ON mn.hotkey = ic.miner_hotkey
WHERE mn.netuid = 64
GROUP BY mn.hotkey, ic.instance_count
ORDER BY min_top_busy_ratio DESC;
"""
