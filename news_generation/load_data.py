import pandas as pd
import json

with open('./data/new例行賽資料_18_24.json', 'r') as f:
    data = json.load(f)

def preprocess_input(input_data):
    # 1. 處理比賽日期與比分
    date = input_data.get("date", "未知日期")
    day = input_data.get("day", "")
    team_away = input_data.get("team_away_name", "客隊")
    team_home = input_data.get("team_home_name", "主隊")
    score_away = input_data.get("score_away", "0")
    score_home = input_data.get("score_home", "0")
    
    game_summary = f"日期:{date} {day}。{team_home} {score_home} 比 {score_away} 擊敗{team_away}。"

    # 2. 處理 MVP 數據
    mvp_data = input_data.get("mvp", {})
    mvp_name = mvp_data.get("球員姓名", "未知球員")
    mvp_stats = mvp_data.get("投球統計") or mvp_data.get("打擊統計", "")
    
    if mvp_data.get("投球統計"):
        innings_pitched = mvp_stats.get("投球局數", "0")
        strikeouts = mvp_stats.get("奪三振數", "0")
        runs_allowed = mvp_stats.get("失分數", "0")
        mvp_summary = f"單場最有價值球員為{mvp_name}，投出{innings_pitched}局，並送出{strikeouts}次三振。"
    else:
        points_get = mvp_stats.get("得分", "0")
        hit_num = mvp_stats.get("安打", "0")
        hr_num = mvp_stats.get("全壘打", "0")
        mvp_summary = f"單場最有價值球員為{mvp_name}，擊出{hit_num}球，和{hr_num}球全壘打，一共獲得{points_get}分。"

    # 3. 處理每局比賽亮點
    all_plays = input_data.get("all_plays", [])
    innings_summary = []
    for inning in all_plays:
        offense_team = inning.get("offense_team", "未知隊伍")
        inning_number = inning.get("inning", "未知局數")
        records = inning.get("records", [])

        # 收集每局的亮點描述
        inning_desc = [f"{inning_number}，進攻隊-{offense_team}："]
        last_point_diff = 0 # 上一棒分差
        has_point = False
        for record in records:
            # player = record.get("player", "未知球員")
            desc = record.get("desc", "無描述")
            away_score = int(record.get("away_score", 0))
            home_score = int(record.get("home_score", 0))
            new_score = abs(away_score - home_score)
            if new_score > last_point_diff:
                has_point = True
                inning_desc.append(f"{desc}")
                inning_desc.append(f"{team_away} {away_score}分：{team_home} {home_score}分")
            last_point_diff = new_score
        # 將每局的描述整理為單段文字
        if has_point :
            innings_summary.append(" ".join(inning_desc))

    # 將所有部分整合
    full_description = f"{game_summary}{mvp_summary} {' '.join(innings_summary)}"
    return full_description

# dic = {"date": [], "title": [], "content": [], "news":[]}
dic = {"content": [], "news":[]}

for d in data:
    date = d['date']
    # dic["date"].append(date)
    # dic["title"].append(d["news"]["title"])

    content = preprocess_input(d)
    dic["content"].append(content)
    if d['news']['content'] is not None:
        dic["news"].append(d["news"]["processed_result"])
    else:
        dic["news"].append(None)

df = pd.DataFrame(dic)
df = df.replace("", pd.NA).dropna()
df = df.reset_index(drop=True)
df.to_csv('./news.csv', index=False)