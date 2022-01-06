from rank import Rank
import argparse
from pyecharts.charts import Bar, Pie, Line
from pyecharts import options as opts

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
    len1, data1 = ydata(dic1)
    len2, data2 = ydata(dic2)
    len3, data3 = ydata(dic3)
    len4, data4 = ydata(dic4)

    pie = (
        Pie()
        .add(
            "活跃用户总数",
            [list(z) for z in zip(['一号楼: ' + str(len1), '二号楼: ' + str(len2), '三号楼: ' + str(len3), '四号楼: ' + str(len4)], [len1, len2, len3, len4])],
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
        .set_global_opts(
            title_opts=opts.TitleOpts(title="专楼发言频率统计"),
            yaxis_opts=opts.AxisOpts(name="用户数"),
            xaxis_opts=opts.AxisOpts(name="发言频次"),
            # toolbox_opts=opts.ToolboxOpts(),
        )
        .overlap(pie)
    )

    return hist

def time_freq():
    dic1 = Rank('statistics/HB_1.json').get_time_seq()
    dic2 = Rank('statistics/HB_2.json').get_time_seq()
    dic3 = Rank('statistics/HB_3.json').get_time_seq()
    dic4 = Rank('statistics/HB_4.json').get_time_seq()
    line = (
        Line()
        .add_xaxis(list(dic1.keys()))
        .add_yaxis("一号楼", list(dic1.values()), linestyle_opts=opts.LineStyleOpts(width=2), is_smooth=True, label_opts=opts.LabelOpts(is_show=False), markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]))
        .add_yaxis("二号楼", list(dic2.values()), linestyle_opts=opts.LineStyleOpts(width=2), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("三号楼", list(dic3.values()), linestyle_opts=opts.LineStyleOpts(width=2), is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("四号楼", list(dic4.values()), linestyle_opts=opts.LineStyleOpts(width=2), is_smooth=True, label_opts=opts.LabelOpts(is_show=False), markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="专楼发言频率统计"),
            yaxis_opts=opts.AxisOpts(
                name="用户数",
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(name="时段"),
            # toolbox_opts=opts.ToolboxOpts(),
        )
    )
    return line

def calen_flow():
    dic1 = Rank('statistics/HB_1.json').get_calen_flow()
    dic2 = Rank('statistics/HB_2.json').get_calen_flow()
    dic3 = Rank('statistics/HB_3.json').get_calen_flow()
    dic4 = Rank('statistics/HB_4.json').get_calen_flow()
    dic = {}
    for k, v in dic1.items():
        if k not in dic:
            dic[k] = v
        else: dic[k] += v
    for k, v in dic2.items():
        if k not in dic:
            dic[k] = v
        else: dic[k] += v
    for k, v in dic3.items():
        if k not in dic:
            dic[k] = v
        else: dic[k] += v
    for k, v in dic4.items():
        if k not in dic:
            dic[k] = v
        else: dic[k] += v
    line = (
        Line()
        .add_xaxis(list(dic.keys())[:-1])
        .add_yaxis("日活跃", list(dic.values())[:-1], linestyle_opts=opts.LineStyleOpts(width=2), is_smooth=True, label_opts=opts.LabelOpts(is_show=False), markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="专楼发言频率统计"),
            yaxis_opts=opts.AxisOpts(
                name="用户数",
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(name="日期"),
            # toolbox_opts=opts.ToolboxOpts(),
        )
    )
    return line

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--act', type=bool, default=False, help='act freq')
    parser.add_argument('--time', type=bool, default=False, help='time freq')
    parser.add_argument('--calen', type=bool, default=False, help='calen flow')
    args = parser.parse_args()
    if args.act == True:
        hist = act_freq()
        hist.render("render/act_freq.html")
    if args.time == True:
        line = time_freq()
        line.render("render/time_freq.html")
    if args.calen == True:
        line = calen_flow()
        line.render("render/calen_flow.html")
