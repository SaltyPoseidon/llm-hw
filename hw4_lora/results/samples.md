## 1. When Essendon played away; where did they play?

схема: `CREATE TABLE table_name_50 (venue VARCHAR, away_team VARCHAR)`

эталон: `SELECT venue FROM table_name_50 WHERE away_team = "essendon"`

- base: `SELECT venue FROM table_name_50 WHERE away_team = 'essendon' AND venue IS NOT NULL;`
- full_ft EM: `SELECT venue FROM table_name_50 WHERE away_team = "essendon"`
- lora EM: `SELECT venue FROM table_name_50 WHERE away_team = "essendon"`

## 2. What is the lowest numbered game against Phoenix with a record of 29-17?

схема: `CREATE TABLE table_name_61 (game INTEGER, opponent VARCHAR, record VARCHAR)`

эталон: `SELECT MIN(game) FROM table_name_61 WHERE opponent = "phoenix" AND record = "29-17"`

- base: `SELECT MIN(game) AS lowest_game FROM table_name_61 WHERE opponent = 'Phoenix' AND record = '29-17';`
- full_ft EM: `SELECT MIN(game) FROM table_name_61 WHERE opponent = "phoenix" AND record = "29-17"`
- lora EM: `SELECT MIN(game) FROM table_name_61 WHERE opponent = "phoenix" AND record = "29-17"`

## 3. What is the transfer fee for Gravgaard?

схема: `CREATE TABLE table_name_94 (transfer_fee VARCHAR, type VARCHAR, name VARCHAR)`

эталон: `SELECT transfer_fee FROM table_name_94 WHERE type = "transfer" AND name = "gravgaard"`

- base: `SELECT transfer_fee FROM table_name_94 WHERE name = 'Gravgaard';`
- full_ft EM: `SELECT transfer_fee FROM table_name_94 WHERE type = "transfer" AND name = "gravgaard"`
- lora EM: `SELECT transfer_fee FROM table_name_94 WHERE type = "transfer" AND name = "gravgaard"`

## 4. How many cup apps for the season where there are more than 10 league goals and 23 league apps?

схема: `CREATE TABLE table_name_17 (cup_apps VARCHAR, league_goals VARCHAR, league_apps VARCHAR)`

эталон: `SELECT cup_apps FROM table_name_17 WHERE league_goals > 10 AND league_apps = 23`

- base: `SELECT COUNT(cup_apps) AS cup_apps_count FROM table_name_17 WHERE league_goals > 10 AND league_apps = 23;`
- full_ft: `SELECT COUNT(cup_apps) FROM table_name_17 WHERE league_goals > 10 AND league_apps = 23`
- lora EM: `SELECT cup_apps FROM table_name_17 WHERE league_goals > 10 AND league_apps = 23`

## 5. Can you tell me the Overall Record that has the Week of 3?

схема: `CREATE TABLE table_name_81 (overall_record VARCHAR, week VARCHAR)`

эталон: `SELECT overall_record FROM table_name_81 WHERE week = 3`

- base: `SELECT overall_record FROM table_name_81 WHERE week = '3';`
- full_ft EM: `SELECT overall_record FROM table_name_81 WHERE week = 3`
- lora EM: `SELECT overall_record FROM table_name_81 WHERE week = 3`

## 6. Of all the contestants who got voted, what is the contestant number and name of the contestant who got least votes?

схема: `CREATE TABLE votes (contestant_number VARCHAR); CREATE TABLE contestants (contestant_number VARCHAR, contestant_name VARCHAR)`

эталон: `SELECT T1.contestant_number, T1.contestant_name FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number = T2.contestant_number GROUP BY T1.contestant_number ORDER BY COUNT(*) LIMIT 1`

- base: `SELECT contestant_number, contestant_name FROM contestants WHERE contestant_number IN ( SELECT contestant_number FROM votes GROUP BY contestant_number ORDER BY COUNT(*) DESC LIMIT 1 );`
- full_ft EM: `SELECT T1.contestant_number, T1.contestant_name FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number = T2.contestant_number GROUP BY T1.contestant_number ORDER BY COUNT(*) LIMIT 1`
- lora: `SELECT T1.contestant_number, T2.contestant_name FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number = T2.contestant_number ORDER BY T2.votes LIMIT 1`

## 7. Which venue had the result 7-1?

схема: `CREATE TABLE table_name_50 (venue VARCHAR, result VARCHAR)`

эталон: `SELECT venue FROM table_name_50 WHERE result = "7-1"`

- base EM: `SELECT venue FROM table_name_50 WHERE result = '7-1';`
- full_ft EM: `SELECT venue FROM table_name_50 WHERE result = "7-1"`
- lora EM: `SELECT venue FROM table_name_50 WHERE result = "7-1"`

## 8. What did the tournament that got an A in 1945 get in 1949?

схема: `CREATE TABLE table_name_22 (Id VARCHAR)`

эталон: `SELECT 1949 FROM table_name_22 WHERE 1945 = "a"`

- base: `SELECT T1.Id FROM table_name_22 AS T1 JOIN ( SELECT Id FROM table_name_22 WHERE Year = 1945 AND Score = 'A' ) AS T2 ON T1.Id = T2.Id WHERE T1.Year = 1949;`
- full_ft EM: `SELECT 1949 FROM table_name_22 WHERE 1945 = "a"`
- lora EM: `SELECT 1949 FROM table_name_22 WHERE 1945 = "a"`

## 9. Find the states where have the colleges whose enrollments are less than the largest size.

схема: `CREATE TABLE college (state VARCHAR, enr INTEGER)`

эталон: `SELECT DISTINCT state FROM college WHERE enr < (SELECT MAX(enr) FROM college)`

- base: `SELECT state FROM college WHERE enr < (SELECT MAX(enr) FROM college);`
- full_ft: `SELECT state FROM college WHERE enr < (SELECT MAX(enr) FROM college)`
- lora: `SELECT state FROM college WHERE enr < (SELECT MAX(enr) FROM college)`

## 10. Which Position has a Competition of world championships, and a Year larger than 1997?

схема: `CREATE TABLE table_name_27 (position VARCHAR, competition VARCHAR, year VARCHAR)`

эталон: `SELECT position FROM table_name_27 WHERE competition = "world championships" AND year > 1997`

- base EM: `SELECT position FROM table_name_27 WHERE competition = 'world championships' AND year > 1997;`
- full_ft EM: `SELECT position FROM table_name_27 WHERE competition = "world championships" AND year > 1997`
- lora EM: `SELECT position FROM table_name_27 WHERE competition = "world championships" AND year > 1997`
