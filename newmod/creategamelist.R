rm(list=ls(all=TRUE))

library('nbaTools')

games = read.csv('gameids.txt',header=FALSE,stringsAsFactors=FALSE, colClasses="character")
colnames(games)[1] = "gameids"

teamnames = read.csv('teamnames.csv',header=FALSE,stringsAsFactors=FALSE)

Vpts = numeric(length=1230)
Hpts = numeric(length=1230)
Vteamname = character(1230)
Hteamname = character(1230)

for (i in c(1:1230))
{
    box = GetBoxScore(GameID = games[i,1])
    tn = box$TEAM_ABBREVIATION

    Vteam = tn[1]
    Hteam = tn[length(tn)]

    Vpts[i] = sum(box$PTS[tn==Vteam])
    Hpts[i] = sum(box$PTS[tn==Hteam])

    Vteamname[i] = teamnames$V1[teamnames$V2==Vteam]
    Hteamname[i] = teamnames$V1[teamnames$V2==Hteam]

    games$away = Vteamname
    games$home = Hteamname
    games$awayscore = Vpts
    games$homescore = Hpts
    write.csv(x=games, file="games.csv", row.names=FALSE)
    save(games, file='games.RData')

    print(i)
    flush.console()
    Sys.sleep(15)
}


