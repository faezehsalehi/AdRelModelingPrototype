SET mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec;
SET mapred.job.priority=NORMAL;

CREATE TABLE faye.encoded_app_categories STORED AS SEQUENCEFILE AS
SELECT
  app_id,
  IF(SUM(IF(category LIKE "%Action%", 1, 0))>=1, 1, 0) as cat_action,
  IF(SUM(IF(category LIKE "%Adventure%", 1, 0))>=1, 1, 0) as cat_adventure,
  IF(SUM(IF(category LIKE "%Card%", 1, 0))>=1, 1, 0) as cat_card,
  IF(SUM(IF(category LIKE "%Casino%", 1, 0))>=1, 1, 0) as cat_casino,
  IF(SUM(IF(category LIKE "%Educational%", 1, 0))>=1, 1, 0) as cat_educational,
  IF(SUM(IF(category LIKE "%Family%", 1, 0))>=1, 1, 0) as cat_family,
  IF(SUM(IF(category LIKE "%Music%", 1, 0))>=1, 1, 0) as cat_music,
  IF(SUM(IF(category LIKE "%Non-Game%", 1, 0))>=1, 1, 0) as cat_non_game,
  IF(SUM(IF(category LIKE "%Puzzle%", 1, 0))>=1, 1, 0) as cat_puzzle,
  IF(SUM(IF(category LIKE "%Racing%", 1, 0))>=1, 1, 0) as cat_racing,
  IF(SUM(IF(category LIKE "%Role%", 1, 0))>=1, 1, 0) as cat_role_playing,
  IF(SUM(IF(category LIKE "%Simulation%", 1, 0))>=1, 1, 0) as cat_simulation,
  IF(SUM(IF(category LIKE "%Sports%", 1, 0))>=1, 1, 0) as cat_sports,
  IF(SUM(IF(category LIKE "%Strategy%", 1, 0))>=1, 1, 0) as cat_strategy,
  IF(SUM(IF(category LIKE "%Trivia%", 1, 0))>=1, 1, 0) as cat_trivia,
  IF(SUM((CASE
    WHEN category LIKE "%Action%" THEN 0
    WHEN category LIKE "%Adventure%" THEN 0
    WHEN category LIKE "%Card%" THEN 0
    WHEN category LIKE "%Casino%" THEN 0
    WHEN category LIKE "%Educational%" THEN 0
    WHEN category LIKE "%Family%" THEN 0
    WHEN category LIKE "%Music%" THEN 0
    WHEN category LIKE "%Non-Game%" THEN 0
    WHEN category LIKE "%Puzzle%" THEN 0
    WHEN category LIKE "%Racing%" THEN 0
    WHEN category LIKE "%Role%" THEN 0
    WHEN category LIKE "%Simulation%" THEN 0
    WHEN category LIKE "%Sports%" THEN 0
    WHEN category LIKE "%Strategy%" THEN 0
    WHEN category LIKE "%Trivia%" THEN 0
    ELSE 1
    END))>=1, 1, 0) as cat_other
  FROM bi.apps_categories
  GROUP BY app_id;


SELECT
  i.advertiser_campaign,
  c.cat_action,
  c.cat_adventure,
  c.cat_card,
  c.cat_casino,
  c.cat_educational,
  c.cat_family,
  c.cat_music,
  c.cat_non_game,
  c.cat_puzzle,
  c.cat_racing,
  c.cat_role_playing,
  c.cat_simulation,
  c.cat_sports,
  c.cat_strategy,
  c.cat_trivia,
  c.cat_other,
  i.country,
  i.region,
  i.reachability,
  i.device_advertising_app_impressions,
  i.bid_value,
  i.ad_type,
  i.media_type,
  i.asset_size,
  i.os,
  i.has_install
FROM warehouse.impressions_enhanced i
  LEFT OUTER JOIN faye.encoded_app_categories c
  ON i.publisher_app=c.app_id
  WHERE i.impression_dt="2017-01-01" and i.advertising_campaign_type=3
limit 100;
