class SwitchNode:
    """
    A class to represent a SwitchNode.
    Attributes
    ----------
    id : int
        The unique identifier for the SwitchNode.
    name : str
        The name of the SwitchNode.
    dwell : float
        The dwell time of the SwitchNode.
    Methods
    -------
    __repr__():
        Returns a string representation of the SwitchNode object.
    """

    def __init__(self, id,start_node, down_start_node,group_id,name,dwell=30):
        self.id = id
        self.start_node = start_node
        self.down_start_node = down_start_node
        self.group_id = group_id #记录道岔所处的组
        self.name = name
        self.dwell = dwell     

    # def sync_switchnodes(self, id,node_id,switch_nodes):
    #     """
    #     Update the SwitchNode with a node_id.
    #     """
    #     self.node_id =node_id
    #     switch_nodes[id-50].node_id = node_id

    def __repr__(self):
        return f"Switch(id={self.id}, name='{self.name}')"
    
    def _get_end_node(self):
        s = int(self.start_node.name[-1]) + 1
        end_node_name = "s1" + str(s)
        return end_node_name

