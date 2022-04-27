from rank import Rank
import numpy as np
import json
import argparse
from pyecharts.charts import Bar, Pie, Line
from pyecharts import options as opts

num2word = {1 : '一', 2 : '二', 3 : '三', 4 : '四', 5 : '五', 6 : '六', 7 : '七', 8 : '八'}

def print_dic(dic):
    for k, v in dic.items():
        print(k, v)

def uid2name(dic):
    with open('uid2name.json', 'r', encoding='utf-8') as f:
        name = json.load(f)
    for k, _ in dic.items():
        if str(k) in name:
            dic[name[str(k)]] = dic[k]
            del dic[k]
    
    return dic

def live(num):
    return Rank('statistics/HB_' + str(num) + '.json').get_live()

def rank(num):
    if num == 0:
        dic = Rank('statistics/HB.json').get_rank()
    else: dic = Rank('statistics/HB_' + str(num) + '.json').get_rank()
    dic = dict(dic[:10])
    return uid2name(dic)

def ydata(dic):
    data = [0 for i in range(8)]
    for ele in dic:
        if ele[0] < 3:
            continue
        if ele[1] > 1000:
            data[7] += 1
        elif ele[1] > 500:
            data[6] += 1
        elif ele[1] > 200:
            data[5] += 1
        elif ele[1] > 50:
            data[4] += 1
        elif ele[1] > 15:
            data[3] += 1
        elif ele[1] > 5:
            data[2] += 1
        elif ele[1] > 1:
            data[1] += 1
        else: data[0] += 1
    return len(dic), data

def act_freq():
    col = ['1', '2-5', '6-15', '16-50', '51-200', '201-500', '501-1000', '1000+']
    dic1 = Rank('statistics/HB_1.json').get_rank()
    dic2 = Rank('statistics/HB_2.json').get_rank()
    dic3 = Rank('statistics/HB_3.json').get_rank()
    dic4 = Rank('statistics/HB_4.json').get_rank()
    dic5 = Rank('statistics/HB_5.json').get_rank()
    dic6 = Rank('statistics/HB_6.json').get_rank()
    len1, data1 = ydata(dic1)
    len2, data2 = ydata(dic2)
    len3, data3 = ydata(dic3)
    len4, data4 = ydata(dic4)
    len5, data5 = ydata(dic5)
    len6, data6 = ydata(dic6)

    pie = (
        Pie()
        .add(
            "活跃用户总数",
            [list(z) for z in zip(['一号楼: ' + str(len1), '二号楼: ' + str(len2), '三号楼: ' + str(len3), '四号楼: ' + str(len4), '五号楼: ' + str(len5), '六号楼: ' + str(len6)], [len1, len2, len3, len4, len5, len6])],
            center=["75%", "35%"],
            radius="40%",
        )
        .set_series_opts(tooltip_opts=opts.TooltipOpts(is_show=True, trigger="item"))
    )

    hist = (
        Bar()
        .add_xaxis(col)
        .add_yaxis('一号楼: ' + str(len1), data1)
        .add_yaxis('二号楼: ' + str(len2), data2)
        .add_yaxis('三号楼: ' + str(len3), data3)
        .add_yaxis('四号楼: ' + str(len4), data4)
        .add_yaxis('五号楼: ' + str(len5), data5)
        .add_yaxis('六号楼: ' + str(len6), data6)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="专楼发言频率"),
            yaxis_opts=opts.AxisOpts(name="用户数"),
            xaxis_opts=opts.AxisOpts(name="发言频次"),
            # toolbox_opts=opts.ToolboxOpts(),
        )
        .overlap(pie)
    )

    return hist

def time_freq(num):
    dic, date_len = Rank('statistics/HB_' + str(num) + '.json').get_time_seq()
    title = num2word[num] + '号楼'
    col = list(dic.keys())[4:] + list(dic.keys())[:4]
    data = list(dic.values())[4:] + list(dic.values())[:4]
    data = np.array(data) / date_len
    data = np.rint(data)
    line = (
        Line()
        .add_xaxis(col)
        .add_yaxis(
            title, data.tolist(),
            linestyle_opts=opts.LineStyleOpts(width=2),
            is_smooth=True, label_opts=opts.LabelOpts(is_show=False),
            markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max"), opts.MarkPointItem(type_="min")]),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="各时段平均水楼数"),
            yaxis_opts=opts.AxisOpts(
                name="楼层数",
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(name="时段"),
            # toolbox_opts=opts.ToolboxOpts(),
        )
    )
    return line

def calen_flow():
    dic = Rank('statistics/HB.json').get_calen_flow()
    line = (
        Line()
        .add_xaxis(list(dic.keys())[:-1])
        .add_yaxis("日活跃", list(dic.values())[:-1], linestyle_opts=opts.LineStyleOpts(width=2), is_smooth=True, label_opts=opts.LabelOpts(is_show=False), markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="专楼日水楼层数"),
            yaxis_opts=opts.AxisOpts(
                name="楼层数",
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(name="日期"),
            # toolbox_opts=opts.ToolboxOpts(),
        )
    )
    return line

def date_act(f=0):
    if f == 0:
        dic = Rank('statistics/HB.json').get_date_act()
    else: dic = Rank('statistics/HB_' + str(args.f) +  '.json').get_date_act()

    line = (
        Line()
        .add_xaxis(list(dic.keys())[1:-1])
        .add_yaxis("日活跃", list(dic.values())[1:-1], linestyle_opts=opts.LineStyleOpts(width=2), is_smooth=True, label_opts=opts.LabelOpts(is_show=False), markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="专楼日活跃用户"),
            yaxis_opts=opts.AxisOpts(
                name="用户数",
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(name="日期"),
            # toolbox_opts=opts.ToolboxOpts(),
        )
    )
    return line

def meme_freq(num):
    dic = Rank('statistics/HB_' + str(num) +  '.json').get_meme_freq()
    dic = dict(dic[:3])
    return dic

def meme_uid(meme, num):
    dic = Rank('statistics/HB_' + str(num) + '.json').get_meme_uid(meme)
    dic = dict(dic[:3])
    return uid2name(dic)

def uid_meme(uid, f=0):
    if f == 0:
        dic = Rank('statistics/HB.json').get_uid_meme(uid)
    else: dic = Rank('statistics/HB_' + str(f) + '.json').get_uid_meme(uid)
    return dict(dic[:3])

def match_reply(match, f=0):
    if f == 0:
        Rank('statistics/HB.json').get_match_reply(match)
    else: Rank('statistics/HB_' + str(f) + '.json').get_match_reply(match)

def uid_reply(uid, f=0):
    if f == 0:
        Rank('statistics/HB.json').get_uid_reply(uid)
    else: Rank('statistics/HB_' + str(f) + '.json').get_uid_reply(uid)

def new_user(num):
    dic = Rank('statistics/HB_' + str(num) +  '.json').get_new_user()
    return dic

def locate(loc, f=0):
    if f == 0:
        Rank('statistics/HB.json').get_locate(loc)
    else: Rank('statistics/HB_' + str(f) + '.json').get_locate(loc)

def ipt(filename):
    Rank(filename + '.json').get_ipt()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=int, default=0, help='HB_f')
    parser.add_argument('--live', type=int, default=0, help='live days')
    parser.add_argument('--rank', type=int, default=-1, help='rank')
    parser.add_argument('--act', type=bool, default=False, help='act freq')
    parser.add_argument('--time', type=int, default=0, help='time freq')
    parser.add_argument('--calen', type=bool, default=False, help='calen flow')
    parser.add_argument('--date', type=bool, default=False, help='date act')
    parser.add_argument('--meme', type=int, default=0, help='meme freq')
    parser.add_argument('--meme_u', type=str, default='', help='meme uid')
    parser.add_argument('--u_meme', type=int, default=0, help='uid meme')
    parser.add_argument('--match', type=str, default='', help='match text')
    parser.add_argument('--uid', type=str, default='', help='uid reply')
    parser.add_argument('--new', type=int, default=0, help='new user')
    parser.add_argument('--ipt', type=str, default='', help='interpretation')
    parser.add_argument('--loc', type=str, default='', help='locate date: mm-dd')
    args = parser.parse_args()
    if args.live != 0:
        print(live(args.live))
    if args.rank != -1:
        dic = rank(args.rank)
        print_dic(dic)
    if args.act == True:
        hist = act_freq()
        hist.render("render/act_freq.html")
    if args.time != 0:
        line = time_freq(args.time)
        line.render("render/time_freq.html")
    if args.calen == True:
        line = calen_flow()
        line.render("render/calen_flow.html")
    if args.date == True:
        line = date_act(args.f)
        line.render("render/date_act.html")
    if args.meme != 0:
        dic = meme_freq(args.meme)
        print_dic(dic)
    if args.meme_u != '':
        dic = meme_uid("\[s:ac:" + args.meme_u[:-1] + "\]", int(args.meme_u[-1]))
        print("[s:ac:" + args.meme_u[:-1] + ']')
        print_dic(dic)
    if args.u_meme != 0:
        dic = uid_meme(args.u_meme, args.f)
        print_dic(dic)
    if args.match != '':
        match_reply(args.match, args.f)
    if args.uid != '':
        uid_reply(args.uid, args.f)
    if args.new != 0:
        dic = new_user(args.new)
        total = np.sum(np.array(list(dic.values())))
        print(total)
        print_dic(dic)
    if args.loc != '':
        locate(args.loc, args.f)
    if args.ipt != '':
        ipt(args.ipt)
