set hive.groupby.orderby.position.alias=true;
SET mapreduce.map.memory.mb=6144;

INSERT OVERWRITE LOCAL DIRECTORY '${hiveconf:OUTPUT_PATH}'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
SELECT
publisher_app,
advertiser_campaign,
advertiser_app,
cat_action,
cat_adventure,
cat_card,
cat_casino,
cat_educational,
cat_family,
cat_music,
cat_non_game,
cat_puzzle,
cat_racing,
cat_role_playing,
cat_simulation,
cat_sports,
cat_strategy,
cat_trivia,
cat_other,
country,
region,
reachability,
device_advertising_app_impressions,
bid_value,
ad_type,
asset_size,
model_id,
has_install
FROM( SELECT
  count(*) over (partition by i.publisher_app) as cnt,
  rank() over (partition by i.publisher_app order by rand()) as rnk,
  i.advertiser_campaign,
  i.advertiser_app,
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
  i.asset_size,
  i.model_id,
  i.has_install
FROM adrel.impressions_enhanced i
  LEFT OUTER JOIN faye.encoded_app_categories c
  ON i.publisher_app=c.app_id
  WHERE i.impression_dt >= '${hiveconf:START_DATE}' and i.impression_dt <= '${hiveconf:END_DATE}'
  and i.advertising_campaign_type=3
  and (i.publisher_app IN (select p.publisher_app from (select count(*) as imps, publisher_app from adrel.impressions_enhanced
       WHERE impression_dt >= '${hiveconf:START_DATE}' and impression_dt <= '${hiveconf:END_DATE}'
       GROUP BY publisher_app ORDER BY imps desc limit 10) p))
  ) mytable
  WHERE rnk <= cnt*0.001;
