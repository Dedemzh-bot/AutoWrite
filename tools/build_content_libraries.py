from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


MATERIAL_GROUPS = {
    "world_stage": ("世界舞台", "world", [
        ("modern_city", "现代都市", ["现实", "都市"], ["一线城市旧城区", "新兴科技城", "沿海旅游城市"], ["隐藏着利益共同体", "正在经历产业洗牌", "表面繁华但阶层割裂", "一场公共事件改变秩序"]),
        ("campus", "校园环境", ["现实", "校园"], ["封闭寄宿高中", "顶尖综合大学", "职业技术学院"], ["排名制度制造对立", "旧案在新生季重启", "社团掌握隐秘资源", "校庆成为冲突爆点"]),
        ("workplace", "职场生态", ["现实", "职场"], ["高速扩张的创业公司", "老牌家族企业", "竞争激烈的专业机构"], ["权责被故意混淆", "关键项目濒临失败", "匿名举报打破平衡", "继任安排引发站队"]),
        ("wealthy_family", "豪门家族", ["现实", "豪门"], ["多房争产的旧豪门", "新贵资本家族", "名望高于财富的世家"], ["继承协议暗藏条件", "婚姻被当作资源交换", "失踪成员突然归来", "家族丑闻面临曝光"]),
        ("ancient_dynasty", "古代王朝", ["古代", "历史"], ["王朝末年的都城", "边疆军镇", "富庶但党争激烈的江南"], ["税制改革触动旧利益", "皇位继承悬而未决", "灾荒迫使地方自救", "外敌与内斗同时升级"]),
        ("xianxia_world", "修仙世界", ["幻想", "修仙"], ["宗门林立的灵气大陆", "飞升通道断绝的末法界", "人妖共治的边境"], ["灵脉正在衰竭", "天道规则出现漏洞", "古老禁区重新开放", "正邪身份被重新定义"]),
        ("apocalypse", "末日废土", ["末日", "生存"], ["病毒爆发后的城市群", "极寒覆盖的避难区", "资源枯竭的荒漠聚落"], ["安全区实行等级配给", "感染机制发生二次变化", "通信恢复带来坏消息", "旧政府设施藏有真相"]),
        ("future_space", "未来星际", ["科幻", "星际"], ["跨星系殖民联盟", "巨型空间站社会", "边境机甲军团"], ["人工智能争取人格权", "跃迁航道突然失效", "基因等级制度遭到挑战", "未知文明留下观察信号"]),
    ]),
    "protagonist": ("主角人设", "character", [
        ("underdog", "低谷逆袭者", ["逆袭", "成长"], ["被行业封杀的前天才", "背负巨债的普通青年", "被家族放弃的继承候选"], ["保留一项无人知晓的能力", "掌握改变局势的旧证据", "必须先隐藏真实目标", "胜利会牺牲重要关系"]),
        ("returning_power", "强者归来", ["回归", "身份"], ["退隐多年的顶级高手", "被宣告死亡的前负责人", "远走他乡的家族弃子"], ["回归只为查清旧案", "力量尚未完全恢复", "旧部已经分裂站队", "必须遵守一份危险承诺"]),
        ("hidden_identity", "隐藏身份", ["身份", "掉马"], ["伪装成新人的行业巨头", "隐姓埋名的权力继承人", "以普通身份生活的传奇人物"], ["身份公开会连累亲友", "对手掌握半真半假的线索", "亲近者误判其动机", "必须在公开场合被迫掉马"]),
        ("reborn", "重生改命者", ["重生", "复仇"], ["回到悲剧发生前的当事人", "带着失败记忆重返少年期", "从终局醒来的边缘角色"], ["前世信息正在快速失效", "改变一件事会引发连锁反应", "真正幕后者与记忆不符", "复仇目标中有人并非恶人"]),
        ("rational_leader", "冷静掌局者", ["智谋", "事业"], ["擅长拆解利益链的谈判者", "习惯记录证据的调查者", "能在混乱中组织团队的领导者"], ["不擅长表达私人情感", "曾因一次判断失误失去同伴", "越接近真相越受制度限制", "必须在效率与底线间选择"]),
        ("contrast_persona", "反差型主角", ["反差", "喜剧"], ["外表柔弱的行动派", "看似散漫的精密策划者", "表面冷淡的共情者"], ["在关键领域极度专业", "日常弱点频繁制造误会", "真实野心与外界评价相反", "只有对手最早看出本质"]),
    ]),
    "supporting_role": ("配角关系", "relationship", [
        ("mentor", "导师与前辈", ["导师", "传承"], ["严苛但守底线的导师", "声名狼藉的前行业传奇", "掌握旧时代秘密的长者"], ["只教方法不替主角收场", "曾与反派共享同一目标", "身体或地位已无法久撑", "留下的考验比答案更重要"]),
        ("rival", "竞争对手", ["竞争", "对照"], ["与主角路线相反的天才", "资源充足的同龄对手", "从朋友变成对手的旧识"], ["尊重实力但不接受手段", "会在共同危机中短暂合作", "背后另有必须获胜的原因", "最终胜负会改变双方价值观"]),
        ("partner", "行动搭档", ["伙伴", "团队"], ["执行力极强的搭档", "善于社交的情报伙伴", "技术能力突出的后勤成员"], ["隐瞒一段与主线有关的过去", "关键时刻会质疑主角决策", "个人目标与团队目标部分冲突", "承担一次不可替代的救场"]),
        ("family", "家庭关系", ["家庭", "亲情"], ["控制欲强的家长", "长期被忽略的手足", "立场复杂的养亲"], ["爱与利益被捆绑在一起", "旧日偏心造成现实后果", "家庭秘密牵连主线证据", "和解必须以边界重建为前提"]),
        ("love_interest", "感情关系方", ["感情", "关系"], ["与主角势均力敌的合作对象", "立场敌对的旧爱", "长期陪伴却被忽略的人"], ["感情推进依赖共同选择", "双方掌握的信息并不对等", "事业目标会制造现实分离", "结局取决于是否尊重边界"]),
        ("antagonist", "核心反派", ["反派", "博弈"], ["相信秩序高于个体的掌权者", "把所有关系视为交易的操盘者", "曾经失败后走向极端的理想主义者"], ["拥有部分合理诉求", "胜利依赖制度而非单纯武力", "与主角共享一段关键经历", "败局会留下新的社会问题"]),
    ]),
    "cheat_device": ("金手指与能力", "device", [
        ("system", "任务系统", ["系统", "成长"], ["发布阶段任务的成长系统", "以声望结算奖励的职业系统", "要求修复剧情偏差的规则系统"], ["奖励伴随明确代价", "任务描述可能故意不完整", "无法替代真实训练与资源", "终极目标与宿主理解不同"]),
        ("space", "随身空间", ["空间", "资源"], ["可储存物资的有限空间", "能培育特殊作物的秘境", "连接废弃基地的私人入口"], ["容量随完成目标扩张", "时间流速存在危险差异", "开启会留下可追踪痕迹", "核心区域需要共同解锁"]),
        ("bloodline", "血脉天赋", ["血脉", "力量"], ["被误判为废脉的稀有血统", "能够感知规则漏洞的天赋", "与古代文明共鸣的基因"], ["使用过度会损伤记忆", "能力成长依赖道德选择", "同源者可能远程感知", "真正用途并非战斗"]),
        ("craft", "专业技艺", ["技能", "职业"], ["近乎失传的修复技艺", "极强的金融建模能力", "能从细节还原现场的观察术"], ["需要长期准备才能发挥", "会暴露主角真实履历", "必须与团队资源结合", "一次误判会带来高额成本"]),
        ("artifact", "关键物品", ["物品", "证据"], ["记录隐藏交易的旧设备", "只能开启一次的传承器物", "来源不明的身份凭证"], ["本身也可能是诱饵", "不同人物能读取不同信息", "使用后会改变所有权", "损坏部分恰好涉及真相"]),
        ("memory", "记忆优势", ["记忆", "信息差"], ["保留前世关键节点的记忆", "拥有他人部分人生片段", "能回忆被集体遗忘的事件"], ["记忆含有被篡改的部分", "越改变未来越失去优势", "无法直接证明其真实性", "关键空白由创伤造成"]),
        ("contract", "契约能力", ["契约", "规则"], ["能交换能力与代价的契约", "约束双方不得说谎的协议", "将承诺转化为现实后果的印记"], ["条款必须双方真实理解", "违约惩罚会波及无辜者", "存在可被利用的定义漏洞", "解除需要放弃最大收益"]),
    ]),
    "plot_event": ("剧情事件", "event", [
        ("public_reversal", "公开反转", ["打脸", "反转"], ["行业评审现场翻盘", "家族会议证据公开", "直播镜头下身份揭晓"], ["反转依据提前埋藏", "胜利后局势继续升级", "旁观者立场发生分裂", "对手仍保留反击筹码"]),
        ("auction", "拍卖与争夺", ["资源", "竞价"], ["地下拍卖出现关键物品", "公开招标争夺核心项目", "遗产竞拍牵出旧案"], ["标的真实价值被误判", "竞价者代表不同势力", "成交只是争夺的开始", "主角必须放弃另一项资源"]),
        ("trial", "考核与比赛", ["考核", "竞技"], ["决定晋升的封闭考核", "跨阵营团队竞赛", "公开展示成果的行业大赛"], ["规则中藏有利益倾向", "竞争者被迫临时合作", "成绩会触发更高层关注", "失败并非淘汰而是调岗"]),
        ("rescue", "救援行动", ["救援", "危机"], ["灾难现场限时救援", "失踪人员追踪行动", "封闭空间突发事故"], ["救援目标隐瞒真实身份", "现场证据与官方说法冲突", "资源只够完成一个方案", "成功会暴露主角底牌"]),
        ("identity_reveal", "身份掉马", ["身份", "掉马"], ["对手公开质疑主角资历", "旧部在众人面前认出主角", "关键权限只能由真实身份开启"], ["掉马解决旧冲突又制造新敌人", "亲近者更在意长期隐瞒", "身份只揭开一层", "公开信息被对手刻意扭曲"]),
        ("investigation", "调查取证", ["悬疑", "证据"], ["追查异常资金流", "复盘多年前的失踪案", "调查被删除的实验记录"], ["证据链缺少关键一环", "证人各自只说部分真话", "合法取证比发现真相更难", "调查者本身成为嫌疑人"]),
        ("separation_return", "分离与重逢", ["关系", "重逢"], ["多年后在竞争项目中重逢", "危机迫使决裂双方再合作", "失踪者以新身份归来"], ["双方掌握的过去版本不同", "重逢并不自动恢复关系", "共同目标结束后仍需选择", "第三方从分离中获得利益"]),
    ]),
    "core_conflict": ("核心冲突", "conflict", [
        ("resource", "资源争夺", ["资源", "利益"], ["稀缺治疗名额之争", "关键技术所有权争议", "生存物资分配冲突"], ["规则表面公平实则偏置", "主角不能只靠出价获胜", "资源背后还有责任义务", "胜者会成为更大目标"]),
        ("identity", "身份与资格", ["身份", "资格"], ["继承资格遭到质疑", "专业资质被恶意撤销", "真实血缘与法律身份冲突"], ["证据存在程序瑕疵", "身份确认会伤害现有关系", "不同制度给出相反结论", "反派利用舆论先行定罪"]),
        ("relationship", "关系边界", ["感情", "边界"], ["长期付出被视为理所当然", "亲密关系被权力控制", "旧承诺与新生活冲突"], ["修复必须先承认实际伤害", "离开会造成现实损失", "第三方并非唯一原因", "选择独立不等于没有感情"]),
        ("institution", "个人与制度", ["制度", "权力"], ["揭露问题会破坏组织声誉", "合法程序保护了错误结果", "改革方案触动多数人利益"], ["制度中也有善意执行者", "个人胜利不能自动改变规则", "证据必须经得起公开审查", "妥协会换来阶段性空间"]),
        ("survival", "生存选择", ["生存", "伦理"], ["撤离名额不足", "感染者是否隔离", "救援路线只能保住一方"], ["信息随时间不断变化", "不选择同样产生后果", "团队价值观出现分裂", "正确方案也需要有人承担代价"]),
        ("truth", "真相与叙事", ["真相", "舆论"], ["官方版本与私人证据冲突", "公众需要的英雄并不存在", "受害者证词彼此矛盾"], ["真相公开可能伤害无辜者", "传播者有自己的利益", "关键证据容易被断章取义", "主角必须选择公开顺序"]),
        ("power", "权力博弈", ["权谋", "阵营"], ["继任者争夺控制权", "地方与中央利益对撞", "联盟内部路线分裂"], ["每次站队都会失去一部分筹码", "中立者掌握关键合法性", "表面敌人可能是临时盟友", "胜利后必须兑现政治承诺"]),
    ]),
    "career_resource": ("职业与资源", "resource", [
        ("medicine", "医疗专业", ["医疗", "专业"], ["急诊团队的分诊经验", "罕见病研究资料", "基层医疗网络"], ["专业判断与行政命令冲突", "样本数量不足以定论", "治疗选择伴随伦理争议", "患者家属掌握关键线索"]),
        ("law", "法律调查", ["法律", "证据"], ["复杂股权诉讼", "刑事案件证据审查", "跨区域执法协作"], ["事实成立但证据不合法", "程序期限制造紧迫感", "证人保护影响公开策略", "和解方案隐藏长期代价"]),
        ("finance", "商业金融", ["商业", "金融"], ["并购中的异常账目", "供应链资金断裂", "家族信托控制权"], ["数字背后是人为设计", "现金流比估值更紧迫", "监管调查突然介入", "关键合同存在双重解释"]),
        ("media", "媒体娱乐", ["娱乐圈", "舆论"], ["危机公关团队", "影视项目制作链", "直播平台流量机制"], ["热搜由多方共同推动", "作品质量与资本安排冲突", "偷拍视频经过恶意剪辑", "粉丝立场影响商业决策"]),
        ("technology", "科技工程", ["科技", "工程"], ["人工智能安全团队", "大型基础设施项目", "前沿生物实验室"], ["技术缺陷无法靠口号修复", "测试数据被选择性披露", "工程期限与安全标准冲突", "核心人员突然离职"]),
        ("military", "军事行动", ["军事", "战略"], ["边境侦察小队", "舰队后勤系统", "城防指挥体系"], ["情报存在时间差", "命令与现场情况冲突", "补给决定战术上限", "胜利必须控制平民代价"]),
    ]),
    "atmosphere": ("场景与氛围", "atmosphere", [
        ("rain_night", "雨夜压迫", ["雨夜", "悬疑"], ["停电后的老城区", "暴雨封锁的山路", "漏水的废弃档案馆"], ["声音比视线更可靠", "通信断续制造误判", "水迹留下关键路线", "黎明前必须做出决定"]),
        ("public_stage", "公开场合", ["公开", "群像"], ["灯光刺眼的发布会", "座无虚席的审判庭", "全网直播的颁奖礼"], ["每个人都在表演立场", "一句话会改变舆论方向", "后台行动与台前发言同步", "沉默被解释为默认"]),
        ("closed_space", "封闭空间", ["封闭", "生存"], ["停运的地铁车厢", "隔离中的高层酒店", "失去动力的飞船"], ["出口规则不断变化", "物资数量可以被核对", "成员身份存在疑点", "时间压力迫使公开秘密"]),
        ("warm_daily", "温暖日常", ["日常", "治愈"], ["清晨营业的小餐馆", "共同修缮的旧房子", "社区节日前的准备"], ["细小照顾体现关系变化", "日常物件承接旧记忆", "平静中埋入下一次选择", "温暖建立在明确边界上"]),
        ("grand_scene", "宏大场面", ["史诗", "高潮"], ["万人见证的决战", "城市级撤离行动", "跨星系舰队会师"], ["个人选择决定全局走向", "胜利画面同时呈现代价", "多个伏笔在同一时刻回收", "新秩序在废墟上建立"]),
    ]),
}


NORMAL_PATTERNS = [
    ("waste_counterattack", "废柴逆袭", "升级", ["逆袭", "成长"], "从被否定到凭真实积累翻盘"),
    ("return_of_king", "王者归来", "身份", ["回归", "掉马"], "隐藏实力逐层揭开并清算旧账"),
    ("hidden_boss", "隐藏大佬", "身份", ["身份", "掉马"], "普通身份与真实权能形成反差"),
    ("system_arrival", "系统降临", "系统", ["系统", "任务"], "任务、奖励与代价推动阶段成长"),
    ("time_travel", "穿越异世", "穿越", ["穿越", "世界"], "利用有限认知适应新制度"),
    ("rebirth_revenge", "重生复仇", "复仇", ["重生", "复仇"], "前世信息优势与蝴蝶效应并行"),
    ("broken_engagement", "退婚逆袭", "关系", ["退婚", "打脸"], "关系决裂后以成长重建价值"),
    ("son_in_law", "赘婿翻身", "身份", ["赘婿", "逆袭"], "低位身份下积蓄资源与话语权"),
    ("prison_return", "出狱归来", "复仇", ["回归", "旧案"], "查清入狱真相并重建生活"),
    ("terminal_countdown", "绝症倒计时", "情感", ["绝症", "选择"], "有限时间迫使人物重新排序人生"),
    ("portable_space", "随身空间经营", "经营", ["空间", "经营"], "资源经营与现实关系同步升级"),
    ("soul_swap", "灵魂互换", "身份", ["互换", "反差"], "借他人身份看见偏见与秘密"),
    ("book_transmigration", "穿书改命", "穿书", ["穿书", "改命"], "已知剧情逐渐失效后的主动选择"),
    ("quick_transmigration", "快穿任务", "任务", ["快穿", "任务"], "单元任务推进总目标与人格变化"),
    ("fake_death", "假死脱身", "悬疑", ["假死", "身份"], "旧身份死亡后重建新局"),
    ("exile_survival", "流放求生", "生存", ["流放", "经营"], "恶劣环境中建立生存共同体"),
    ("bankruptcy_rebuild", "破产重来", "事业", ["破产", "创业"], "从资源清零到重建信用"),
    ("public_face_slap", "公开打脸", "爽点", ["打脸", "证据"], "先建立对方确信再用证据翻盘"),
    ("identity_reveal", "身份掉马", "身份", ["掉马", "反转"], "身份揭晓改变关系和权力结构"),
    ("evidence_counter", "证据反杀", "悬疑", ["证据", "反杀"], "完整证据链在关键场合闭环"),
    ("heroic_rescue", "危机救援", "行动", ["救援", "危机"], "能力在高压行动中被验证"),
    ("misunderstanding_truth", "误会解开", "关系", ["误会", "真相"], "信息来源、选择与伤害逐层澄清"),
    ("multiple_vests", "多重马甲", "身份", ["马甲", "掉马"], "不同身份分别承担功能并有序揭晓"),
    ("secret_realm", "秘境夺宝", "冒险", ["秘境", "资源"], "规则探索、资源争夺与队伍博弈"),
    ("auction_war", "拍卖争夺", "商业", ["拍卖", "资源"], "竞价背后是情报和势力较量"),
    ("tournament", "比武扬名", "竞技", ["竞技", "升级"], "每轮对手验证不同能力短板"),
    ("assessment", "考核晋级", "竞技", ["考核", "成长"], "制度化关卡推动身份跃迁"),
    ("family_banquet", "家族宴会翻盘", "豪门", ["家族", "打脸"], "公开场合完成关系与利益重排"),
    ("conspiracy_break", "阴谋破局", "权谋", ["阴谋", "证据"], "从异常结果逆推利益链"),
    ("desperate_counterkill", "绝境反杀", "行动", ["绝境", "反杀"], "依靠前置资源在极限条件翻盘"),
    ("undercover", "卧底潜伏", "悬疑", ["卧底", "身份"], "双重身份与信任成本持续升级"),
    ("kinship_reveal", "认亲风波", "家庭", ["认亲", "身份"], "血缘、法律和情感归属发生冲突"),
    ("amnesia", "失忆追真", "悬疑", ["失忆", "真相"], "用证据而非直觉重建过去"),
    ("substitute_love", "替身觉醒", "情感", ["替身", "觉醒"], "停止扮演他人后重建自我"),
    ("wife_chasing", "追妻火葬场", "情感", ["追妻", "悔悟"], "伤害、离开、代价和边界依次成立"),
    ("husband_chasing", "追夫火葬场", "情感", ["追夫", "悔悟"], "性转关系中的伤害与追悔闭环"),
    ("contract_love", "契约恋爱", "情感", ["契约", "恋爱"], "规则边界在共同经历中改变"),
    ("marriage_first", "先婚后爱", "情感", ["婚姻", "成长"], "从利益合作到可见信任积累"),
    ("second_chance", "破镜重圆", "情感", ["重逢", "修复"], "先解决旧伤再决定是否重建"),
    ("true_fake_heir", "真假千金", "身份", ["千金", "身份"], "身份错位引发家庭与资源重排"),
    ("runaway_pregnancy", "带球跑", "情感", ["亲子", "分离"], "隐瞒原因、责任与边界必须合理"),
    ("cute_child", "萌宝助攻", "家庭", ["萌宝", "亲情"], "孩子推动关系但不替成人做决定"),
    ("xianxia_progression", "修仙升级", "升级", ["修仙", "升级"], "境界、资源、地图和责任同步扩张"),
    ("fallen_genius", "天才陨落再起", "升级", ["天才", "逆袭"], "失去旧优势后重建能力体系"),
    ("inheritance_trial", "传承试炼", "冒险", ["传承", "试炼"], "获得力量必须通过价值选择"),
    ("sect_rise", "宗门崛起", "经营", ["宗门", "经营"], "人才、资源、制度和外敌共同推进"),
    ("demon_conversion", "正魔立场反转", "修仙", ["正邪", "反转"], "阵营标签与真实行为形成冲突"),
    ("urban_doctor", "都市神医", "都市", ["医术", "逆袭"], "专业救治带出利益链与身份线"),
    ("veteran_return", "兵王归来", "都市", ["兵王", "回归"], "旧能力进入现代秩序后的克制使用"),
    ("business_empire", "商业帝国", "事业", ["商业", "成长"], "产品、现金流、团队和资本博弈"),
    ("entertainment_rise", "娱乐圈逆袭", "事业", ["娱乐圈", "掉马"], "作品能力与舆论资本双线推进"),
    ("official_career", "官场沉浮", "权谋", ["官场", "制度"], "程序、民意、利益和责任相互制衡"),
    ("historical_hegemony", "历史争霸", "历史", ["争霸", "权谋"], "地盘、财政、军队和合法性共同扩张"),
    ("court_intrigue", "宫斗夺权", "古言", ["宫斗", "权谋"], "信息、人心和制度决定胜负"),
    ("house_intrigue", "宅斗自救", "古言", ["宅斗", "经营"], "财产、名分与家族规则逐步破局"),
    ("farming_management", "种田经营", "经营", ["种田", "经营"], "生产、交易、社区关系形成增长循环"),
    ("apocalypse_survival", "末世求生", "末日", ["末日", "生存"], "物资、规则、队伍和人性持续受压"),
    ("base_building", "基地建设", "末日", ["基地", "经营"], "安全、生产、人口与制度同步建设"),
    ("interstellar_mecha", "星际机甲", "科幻", ["机甲", "军校"], "训练、战斗、团队和文明危机升级"),
    ("rule_horror", "规则怪谈", "悬疑", ["规则", "生存"], "规则验证、真假辨别与代价闭环"),
    ("infinite_trials", "无限副本", "悬疑", ["副本", "闯关"], "单元关卡与总谜团同步推进"),
]


STRONG_SPECS = [
    ("female_angst_awakening", "女频虐恋觉醒", "情感", ["虐恋", "觉醒"], ["cheat_device"], ["hot_blood", "humor"], ["default", "literary", "cold", "emotional_tension", "realist_ensemble", "ancient_elegant"], {"no_reunion": "主角独立离开且不复合", "costly_reunion": "伤害方付出长期代价后由主角决定是否重启关系"}),
    ("male_angst_awakening", "虐恋觉醒性转", "情感", ["虐恋", "觉醒"], ["cheat_device"], ["hot_blood", "humor"], ["default", "literary", "cold", "emotional_tension", "realist_ensemble", "ancient_elegant"], {"no_reunion": "男主独立离开且不复合", "costly_reunion": "女主付出长期代价后由男主决定是否重启关系"}),
    ("strong_rule_horror", "强规则怪谈", "悬疑", ["规则", "生存"], ["love_interest"], ["甜宠", "无代价外挂"], ["default", "cold", "literary", "suspense", "realist_ensemble"], {"escape_truth": "破解核心规则并带着真相逃离", "contain_source": "付出代价封存污染源", "become_rule": "幸存但成为新规则的一部分"}),
    ("strong_historical_power", "历史权谋强套路", "历史", ["权谋", "制度"], ["cheat_device"], ["无成本现代碾压"], ["default", "cold", "literary", "ancient_elegant", "realist_ensemble"], {"claim_legitimacy": "取得合法性并重排权力格局", "retire_after_reform": "完成改革后退出权力中心", "tragic_balance": "以个人代价换取阶段平衡"}),
    ("strong_male_power_progression", "男频升级打脸强套路", "升级", ["升级", "打脸"], ["love_interest"], ["纯虐恋"], ["default", "hot_blood", "cold", "humor", "business"], {"claim_throne": "完成阶段登顶并掌握新权柄", "open_bigger_map": "击败当前压迫者并打开更高地图", "protect_brothers": "赢下关键战并建立自己的班底"}),
    ("strong_system_progression", "系统任务升级强套路", "系统", ["系统", "任务"], ["atmosphere"], ["无代价奖励"], ["default", "hot_blood", "humor", "suspense"], {"break_system": "识破系统终极目的并获得自主权", "master_system": "完成终局任务并重写系统规则", "pay_the_price": "以明确代价换取最终奖励"}),
    ("strong_xuanhuan_map_progression", "玄幻升级换地图强套路", "升级", ["玄幻", "升级"], ["love_interest"], ["纯日常"], ["default", "hot_blood", "cold", "ancient_elegant"], {"ascend_realm": "完成大境界突破进入更高世界", "found_sect": "建立自己的势力与传承", "seal_catastrophe": "付出代价封印跨界灾难"}),
    ("strong_apocalypse_base", "末世基地生存强套路", "末日", ["末日", "基地"], ["love_interest"], ["无资源压力"], ["default", "cold", "hot_blood", "suspense", "realist_ensemble"], {"stable_base": "建立可持续基地与公开制度", "migrate_survivors": "带领幸存者迁往新安全区", "cure_with_cost": "获得阶段解药但承担长期代价"}),
    ("strong_infinite_dungeon", "无限副本闯关强套路", "悬疑", ["副本", "规则"], ["career_resource"], ["无规则闯关"], ["default", "cold", "literary", "suspense"], {"return_reality": "破解主系统返回现实", "free_players": "摧毁控制核心释放参与者", "new_gatekeeper": "成为守门人并改变副本规则"}),
    ("strong_ancient_house_revenge", "古言宅斗复仇强套路", "古言", ["宅斗", "复仇"], ["cheat_device"], ["无证据打脸"], ["default", "literary", "cold", "ancient_elegant", "emotional_tension"], {"independent_household": "脱离旧家族建立独立生活", "restore_name": "恢复名誉并夺回合法财产", "enter_power_center": "借宅斗成果进入更大权力局"}),
    ("strong_true_fake_heir", "真假千金身份逆袭强套路", "身份", ["千金", "身份"], ["cheat_device"], ["强行团宠"], ["default", "literary", "cold", "emotional_tension", "realist_ensemble", "sweet_romcom"], {"leave_family": "确认身份后仍选择离开失衡家庭", "rebuild_family": "在责任清算后重建有限关系", "career_independence": "用事业与新关系完成独立"}),
    ("strong_entertainment_reveal", "娱乐圈掉马逆袭强套路", "事业", ["娱乐圈", "掉马"], ["world_stage"], ["空降热搜翻盘"], ["default", "humor", "hot_blood", "business", "sweet_romcom"], {"award_reveal": "用作品成绩与身份揭晓完成翻盘", "studio_independence": "脱离旧资本建立独立团队", "expose_chain": "公开证据摧毁造谣利益链"}),
    ("strong_interstellar_mecha", "星际机甲军校成长强套路", "科幻", ["机甲", "军校"], ["love_interest"], ["单兵无团队"], ["default", "hot_blood", "cold", "suspense"], {"fleet_command": "赢得关键战并获得舰队指挥权", "academy_reform": "揭露制度问题并推动军校改革", "civilization_gate": "守住边境并打开文明新阶段"}),
]


def build_material_library() -> dict:
    groups = {}
    entries = []
    for group_id, (group_name, default_slot, children) in MATERIAL_GROUPS.items():
        groups[group_id] = {
            "name": group_name,
            "default_slot": default_slot,
            "subcategories": [],
        }
        for child_id, child_name, tags, cores, twists in children:
            groups[group_id]["subcategories"].append({
                "id": child_id,
                "name": child_name,
                "tags": tags,
            })
            index = 0
            for core in cores:
                for twist in twists:
                    index += 1
                    entries.append({
                        "id": f"M-{group_id[:3].upper()}-{child_id.upper()}-{index:02d}",
                        "text": f"{core}，{twist}",
                        "category": group_id,
                        "subcategory": child_id,
                        "slot": default_slot,
                        "tags": list(dict.fromkeys(tags)),
                        "drivers": list(dict.fromkeys(tags)),
                    })
    return {
        "schema_version": 2,
        "groups": groups,
        "slot_order": ["character", "conflict", "world", "device", "relationship", "event", "resource", "atmosphere"],
        "entries": entries,
    }


def _normal_pattern(item: tuple) -> dict:
    key, name, category, tags, focus = item
    return {
        "id": key,
        "name": name,
        "category": category,
        "strong": False,
        "tags": tags,
        "architect": f"以“{focus}”为主线设计完整因果链，每次兑现都必须改变局势，避免只重复同一种桥段。",
        "writer": f"本章如轮到该套路推进，应通过具体行动兑现“{focus}”，不得用旁白直接宣布结果。",
        "auditor": f"检查“{focus}”是否有前因、行动、结果和后续影响；辅助使用时缺失仅作警告。",
        "hard_conflicts": [],
        "soft_conflicts": [],
        "forbidden_material_categories": [],
        "forbidden_material_tags": [],
        "soft_material_tags": [],
        "compatible_styles": [],
        "ending_options": {},
    }


def _strong_pattern(spec: tuple) -> dict:
    key, name, category, tags, forbidden_categories, forbidden_tags, styles, endings = spec
    focus = "、".join(tags)
    return {
        "id": key,
        "name": name,
        "category": category,
        "strong": True,
        "tags": tags,
        "architect": f"严格按{focus}主驱动力规划全书；每一阶段必须改变资源、关系、身份或规则，禁止原地重复。",
        "writer": f"必须完成本章结构化任务，通过场景、行动和后果推进{focus}，不能用总结代替剧情。",
        "auditor": f"逐项检查本章强制任务、主角状态变化和{focus}因果闭环；缺失必须退稿。",
        "hard_conflicts": [],
        "soft_conflicts": [],
        "forbidden_material_categories": forbidden_categories,
        "forbidden_material_tags": forbidden_tags,
        "soft_material_tags": [],
        "compatible_styles": styles,
        "ending_options": endings,
        "material_note": "素材只能补充舞台、人物、资源和局部冲突，不得替代主套路核心驱动力。",
        "manifest_labels": {
            "protagonist": "主角",
            "counterpart": "核心对手或关系方",
            "foil": "对照人物",
            "conflict": "核心冲突",
            "ending": "结局",
        },
        "protagonist_pool": [
            f"在{focus}压力下仍坚持底线的行动者",
            f"曾因误判付出代价、必须重新学习{focus}规则的主角",
            f"资源有限但擅长观察和组织的成长型主角",
        ],
        "counterpart_pool": [
            f"掌握主要资源并代表旧有{focus}秩序的对手",
            "与主角目标相似但手段相反的竞争者",
            "能借制度和舆论扩大压力的关键关系方",
        ],
        "foil_pool": [
            "前期轻视主角、后期被迫重新站队的见证者",
            "提供错误路线并承担后果的对照人物",
            "在共同危机中暴露真实立场的临时盟友",
        ],
        "background_pool": [
            f"高压{category}环境",
            f"资源紧缺的{category}舞台",
            f"规则正在变化的{category}社会",
            f"多方势力交错的{category}核心区",
        ],
        "conflict_modules": [
            {"id": "resource_lock", "name": "资源封锁", "category": "resource", "description": "关键资源被对手控制，主角必须建立替代路径。"},
            {"id": "public_pressure", "name": "公开压力", "category": "reputation", "description": "公开场合的质疑迫使主角用行动和证据回应。"},
            {"id": "false_information", "name": "错误情报", "category": "information", "description": "半真半假的信息诱导主角付出阶段代价。"},
            {"id": "team_split", "name": "团队分裂", "category": "relationship", "description": "价值冲突导致队伍分裂，必须重新建立合作条件。"},
            {"id": "rule_escalation", "name": "规则升级", "category": "rule", "description": "原有策略失效，主角必须理解更深层机制。"},
            {"id": "irreversible_cost", "name": "不可逆代价", "category": "cost", "description": "关键选择造成无法轻易撤销的损失并影响终局。"},
        ],
        "beats": [
            {"id": "hook", "range": "前300字", "requirement": f"立即爆发{focus}核心压力并给出可追踪目标。"},
            {"id": "accumulation", "range": "0%-25%", "requirement": "连续验证规则、资源或关系限制，主角开始积累有效筹码。"},
            {"id": "escalation", "range": "25%-45%", "requirement": "对手升级压力，既有选择产生不可忽略的后果。"},
            {"id": "paywall_turn", "range": "45%-50%", "requirement": "完成改变行动策略的核心反转，旧方案彻底失效。"},
            {"id": "truth_regret", "range": "50%-80%", "requirement": "真相与代价逐层兑现，主角开始主动重排局势。"},
            {"id": "independent_ending", "range": "80%-100%", "requirement": "回收核心伏笔并兑现已确认结局，禁止临时开启新主线。"},
        ],
        "writing_techniques": [
            "每次升级或反转都必须来自已出现的能力、证据、资源或关系。",
            "用行动结果展示地位变化，不用旁白宣布主角变强。",
            "同一种冲突不得连续重复，必须提高层级或改变性质。",
            "阶段胜利必须带出新的责任、敌人或代价。",
        ],
        "audit_rules": [
            "本章结构化任务未完成时判定不通过。",
            "核心反转必须改变后续策略，只有惊讶效果不算完成。",
            "主角获得关键能力或资源必须有来源和代价。",
            "辅助套路不得覆盖主套路结局和人物弧线。",
        ],
        "forbidden": [
            "禁止敌人无理由降智送资源。",
            "禁止临时增加未铺垫的万能能力。",
            "禁止胜利后局势、关系和资源完全不变。",
        ],
    }


def build_pattern_library() -> dict:
    patterns = {
        "none": {
            "id": "none",
            "name": "无固定主套路",
            "category": "自由",
            "strong": False,
            "tags": [],
            "architect": "按用户设定和人物因果自然组织故事。",
            "writer": "按大纲和人物动机自然推进。",
            "auditor": "只检查大纲、逻辑、连续性和结局完整性。",
            "hard_conflicts": [],
            "soft_conflicts": [],
            "forbidden_material_categories": [],
            "forbidden_material_tags": [],
            "soft_material_tags": [],
            "compatible_styles": [],
            "ending_options": {},
        },
        "custom": {
            "id": "custom",
            "name": "自定义主套路",
            "category": "自定义",
            "strong": False,
            "tags": ["自定义"],
            "architect": "提炼用户提供的主线、阶段目标、必备桥段和禁忌。",
            "writer": "执行用户确认的自定义要求，但不得破坏大纲和连续性。",
            "auditor": "检查自定义要求是否兑现；无法确定的偏好只作警告。",
            "hard_conflicts": [],
            "soft_conflicts": [],
            "forbidden_material_categories": [],
            "forbidden_material_tags": [],
            "soft_material_tags": [],
            "compatible_styles": [],
            "ending_options": {},
        },
    }
    for item in NORMAL_PATTERNS:
        patterns[item[0]] = _normal_pattern(item)
    for spec in STRONG_SPECS:
        patterns[spec[0]] = _strong_pattern(spec)

    hard_pairs = [
        ("female_angst_awakening", "strong_male_power_progression"),
        ("male_angst_awakening", "strong_male_power_progression"),
        ("strong_rule_horror", "marriage_first"),
        ("strong_rule_horror", "cute_child"),
        ("strong_historical_power", "interstellar_mecha"),
        ("strong_apocalypse_base", "warm_daily"),
        ("strong_infinite_dungeon", "farming_management"),
    ]
    for left, right in hard_pairs:
        if left in patterns and right in patterns:
            patterns[left]["hard_conflicts"].append(right)
            patterns[right]["hard_conflicts"].append(left)

    return {
        "schema_version": 2,
        "patterns": patterns,
        "structure_templates": {
            "three_act": ["建立目标与困境", "升级对抗并完成核心反转", "承担代价并完成结局"],
            "golden_three": ["首章立即爆发冲突", "第二章扩大信息差与目标", "第三章兑现第一次有效反转"],
            "quest_loop": ["接受目标", "遭遇限制", "付出行动", "结算结果", "引出更高层目标"],
            "progression_map": ["当前层级受压", "积累资源能力", "公开突破", "重排势力", "进入更高地图"],
        },
    }


def write_json(name: str, payload: dict) -> None:
    path = ROOT / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{name}: {path.stat().st_size} bytes")


def main() -> None:
    materials = build_material_library()
    patterns = build_pattern_library()
    if len(materials["entries"]) < 500:
        raise RuntimeError("素材不足500条")
    normal_count = sum(
        not item.get("strong") and key not in {"none", "custom"}
        for key, item in patterns["patterns"].items()
    )
    strong_count = sum(item.get("strong") for item in patterns["patterns"].values())
    if normal_count < 60 or strong_count != 13:
        raise RuntimeError(f"套路数量不正确：normal={normal_count}, strong={strong_count}")
    write_json("material_library.json", materials)
    write_json("pattern_library.json", patterns)
    print(f"materials={len(materials['entries'])}, normal={normal_count}, strong={strong_count}")


if __name__ == "__main__":
    main()
