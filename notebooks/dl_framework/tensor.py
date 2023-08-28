import numpy as np

class Tensor:
    def __init__(self, data, autograd=False, creators=None, creation_op=None, _id=None):
        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.id = _id

        self.children = {}
        if _id is None:
            _id = np.random.randint(0, 100000)
        self.id = _id

        if creators is not None:
            for t in self.creators:
                if self.id not in t.children:
                    t.children[self.id] = 1
                else:
                    t.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for _id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
            if self.creation_op == "add":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)
            elif self.creation_op == "neg":
                self.creators[0].backward(self.grad.__neg__())
            elif self.creation_op == "sub":
                t = Tensor(self.grad.data)
                self.creators[0].backward(t, self)
                t = Tensor(self.grad.__neg__().data)
                self.creators[1].backward(t, self)
            elif self.creation_op == "mul":
                t = self.grad * self.creators[1]
                self.creators[0].backward(t, self)
                t = self.grad * self.creators[0]
                self.creators[1].backward(t, self)
            elif self.creation_op == "mm":
                act = self.creators[0]
                weights = self.creators[1]
                t = self.grad.mm(weights.transpose())
                act.backward(t)
                t = self.grad.transpose().mm(act).transpose()
                weights.backward(t)
            elif self.creation_op == "transpose":
                self.creators[0].backward(self.grad.transpose())
            elif "sum" in self.creation_op:
                dim = int(self.creation_op.split("_")[1])
                ds = self.creators[0].data.shape[dim]
                self.creators[0].backward(self.grad.expand(dim, ds))
            elif "expand" in self.creation_op:
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.sum(dim))
            elif self.creation_op == "sigmoid":
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (self * (ones - self)))
            elif self.creation_op == "tanh":
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (ones - (self * self)))
            elif self.creation_op == "index_select":
                new_grad = np.zeros_like(self.creators[0].data)
                indices_ = self.index_select_indices.data.flatten()
                grad_ = grad.data.reshape(len(indices_), -1)
                for i in range(len(indices_)):
                    new_grad[indices_[i]] += grad_[i]
                self.creators[0].backward(Tensor(new_grad))
            elif self.creation_op == "cross_entropy":
                dx = self.softmax_output - self.target_dist
                self.creators[0].backward(Tensor(dx))

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd=True, creators=[self, other], creation_op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, autograd=True, creators=[self], creation_op="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, autograd=True, creators=[self, other], creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, autograd=True, creators=[self, other], creation_op="mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim), autograd=True, creators=[self], creation_op="sum_" + str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        # trans_cmd = list(range(0, len(self.data.shape)))
        # trans_cmd.insert(dim, len(self.data.shape))
        # new_shape = list(self.data.shape) + [copies]
        # new_data = self.data.repeat(copies).reshape(new_shape)
        # new_data = new_data.transpose(trans_cmd)
        # if self.autograd:
        #     return Tensor(new_data, autograd=True, creators=[self], creation_op="expand_" + str(dim))
        # return Tensor(new_data)
        if self.autograd:
            return Tensor(np.expand_dims(self.data, dim).repeat(copies, axis=dim), autograd=True, creators=[self], creation_op="expand_" + str(dim))
        return Tensor(np.expand_dims(self.data, dim).repeat(copies, axis=dim))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), autograd=True, creators=[self], creation_op="transpose")
        return Tensor(self.data.transpose())

    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data), autograd=True, creators=[self, x], creation_op="mm")
        return Tensor(self.data.dot(x.data))

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)), autograd=True, creators=[self], creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), autograd=True, creators=[self], creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def index_select(self, indices):
        if self.autograd:
            new = Tensor(self.data[indices.data], autograd=True, creators=[self], creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp, axis=len(self.data.shape)-1, keepdims=True)
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * target_dist).sum(1).mean()
        if self.autograd:
            out = Tensor(loss, autograd=True, creators=[self], creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out
        return Tensor(loss)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
