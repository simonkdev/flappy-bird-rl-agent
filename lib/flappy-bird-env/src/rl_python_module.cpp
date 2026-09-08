#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <new>
#include <optional>
#include <stdexcept>

#include <rl/FlappyEnv.hpp>

namespace {
struct NativeFlappyEnvObject {
	PyObject_HEAD
	FlappyEnv* env = nullptr;
};

PyObject* exceptionToPyErr() {
	try {
		throw;
	} catch (const std::exception& e) {
		PyErr_SetString(PyExc_RuntimeError, e.what());
	} catch (...) {
		PyErr_SetString(PyExc_RuntimeError, "Unknown native FlappyEnv error");
	}
	return nullptr;
}

std::optional<unsigned int> optionalSeedFromPy(PyObject* seedObject) {
	if (!seedObject || seedObject == Py_None) {
		return std::nullopt;
	}

	const unsigned long seed = PyLong_AsUnsignedLong(seedObject);
	if (PyErr_Occurred()) {
		throw std::invalid_argument("seed must be a non-negative integer or None");
	}
	return static_cast<unsigned int>(seed);
}

PyObject* resultToTuple(const FlappyEnvStepResult& result, const int width, const int height) {
	const auto& observation = *result.observation;
	PyObject* observationBytes = PyBytes_FromStringAndSize(
		reinterpret_cast<const char*>(observation.data()),
		static_cast<Py_ssize_t>(observation.size()));
	if (!observationBytes) {
		return nullptr;
	}

	PyObject* tuple = PyTuple_New(10);
	if (!tuple) {
		Py_DECREF(observationBytes);
		return nullptr;
	}

	PyTuple_SET_ITEM(tuple, 0, observationBytes);
	PyTuple_SET_ITEM(tuple, 1, PyLong_FromLong(width));
	PyTuple_SET_ITEM(tuple, 2, PyLong_FromLong(height));
	PyTuple_SET_ITEM(tuple, 3, PyUnicode_FromString("uint8"));
	PyTuple_SET_ITEM(tuple, 4, PyFloat_FromDouble(result.reward));
	PyTuple_SET_ITEM(tuple, 5, PyBool_FromLong(result.terminated));
	PyTuple_SET_ITEM(tuple, 6, PyBool_FromLong(result.alive));
	PyTuple_SET_ITEM(tuple, 7, PyLong_FromLong(result.score));
	PyTuple_SET_ITEM(tuple, 8, PyBool_FromLong(result.passedPipe));
	PyTuple_SET_ITEM(tuple, 9, PyFloat_FromDouble(result.simulationTime));

	for (Py_ssize_t i = 0; i < 10; ++i) {
		if (!PyTuple_GET_ITEM(tuple, i)) {
			Py_DECREF(tuple);
			return nullptr;
		}
	}

	return tuple;
}

int NativeFlappyEnv_init(NativeFlappyEnvObject* self, PyObject* args, PyObject* kwargs) {
	int width = 42;
	int height = 42;
	int ticksPerStep = 4;
	double dt = 1.0 / 60.0;
	PyObject* seedObject = Py_None;
	int debugWindow = 0;
	int showGameWindow = 0;

	static const char* keywords[] = {
		"width",
		"height",
		"ticks_per_step",
		"dt",
		"seed",
		"debug_window",
		"show_game_window",
		nullptr
	};

	if (!PyArg_ParseTupleAndKeywords(
		    args,
		    kwargs,
		    "|iiidOpp",
		    const_cast<char**>(keywords),
		    &width,
		    &height,
		    &ticksPerStep,
		    &dt,
		    &seedObject,
		    &debugWindow,
		    &showGameWindow)) {
		return -1;
	}

	try {
		FlappyEnvConfig config;
		config.observationWidth = width;
		config.observationHeight = height;
		config.ticksPerStep = ticksPerStep;
		config.fixedDeltaTime = static_cast<float>(dt);
		config.seed = optionalSeedFromPy(seedObject);
		config.debugWindow = debugWindow;
		config.showGameWindow = showGameWindow;
		config.resourceRoot = FLAPPY_BIRD_RESOURCE_DIR;
		self->env = new FlappyEnv(config);
		return 0;
	} catch (...) {
		exceptionToPyErr();
		return -1;
	}
}

void NativeFlappyEnv_dealloc(NativeFlappyEnvObject* self) {
	delete self->env;
	self->env = nullptr;
	Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* NativeFlappyEnv_reset(NativeFlappyEnvObject* self, PyObject* args, PyObject* kwargs) {
	PyObject* seedObject = Py_None;
	static const char* keywords[] = {"seed", nullptr};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", const_cast<char**>(keywords), &seedObject)) {
		return nullptr;
	}

	try {
		const FlappyEnvStepResult result = self->env->reset(optionalSeedFromPy(seedObject));
		return resultToTuple(result, self->env->observationWidth(), self->env->observationHeight());
	} catch (...) {
		return exceptionToPyErr();
	}
}

PyObject* NativeFlappyEnv_step(NativeFlappyEnvObject* self, PyObject* args) {
	int action = 0;
	if (!PyArg_ParseTuple(args, "i", &action)) {
		return nullptr;
	}

	try {
		FlappyEnvStepResult result;
		Py_BEGIN_ALLOW_THREADS
		result = self->env->step(action);
		Py_END_ALLOW_THREADS
		return resultToTuple(result, self->env->observationWidth(), self->env->observationHeight());
	} catch (...) {
		return exceptionToPyErr();
	}
}

PyObject* NativeFlappyEnv_info(NativeFlappyEnvObject* self, PyObject*) {
	return Py_BuildValue(
		"(iif)",
		self->env->observationWidth(),
		self->env->observationHeight(),
		static_cast<double>(self->env->fixedDeltaTime()));
}

PyObject* NativeFlappyEnv_close(NativeFlappyEnvObject* self, PyObject*) {
	delete self->env;
	self->env = nullptr;
	Py_RETURN_NONE;
}

PyMethodDef NativeFlappyEnv_methods[] = {
	{"reset", reinterpret_cast<PyCFunction>(NativeFlappyEnv_reset), METH_VARARGS | METH_KEYWORDS, nullptr},
	{"step", reinterpret_cast<PyCFunction>(NativeFlappyEnv_step), METH_VARARGS, nullptr},
	{"info", reinterpret_cast<PyCFunction>(NativeFlappyEnv_info), METH_NOARGS, nullptr},
	{"close", reinterpret_cast<PyCFunction>(NativeFlappyEnv_close), METH_NOARGS, nullptr},
	{nullptr, nullptr, 0, nullptr},
};

PyTypeObject NativeFlappyEnvType = {
	PyVarObject_HEAD_INIT(nullptr, 0)
};

PyModuleDef nativeFlappyEnvModule = {
	PyModuleDef_HEAD_INIT,
	"_native_flappy_env",
	nullptr,
	-1,
	nullptr,
};
}

PyMODINIT_FUNC PyInit__native_flappy_env() {
	NativeFlappyEnvType.tp_name = "_native_flappy_env.NativeFlappyEnv";
	NativeFlappyEnvType.tp_basicsize = sizeof(NativeFlappyEnvObject);
	NativeFlappyEnvType.tp_itemsize = 0;
	NativeFlappyEnvType.tp_flags = Py_TPFLAGS_DEFAULT;
	NativeFlappyEnvType.tp_new = PyType_GenericNew;
	NativeFlappyEnvType.tp_init = reinterpret_cast<initproc>(NativeFlappyEnv_init);
	NativeFlappyEnvType.tp_dealloc = reinterpret_cast<destructor>(NativeFlappyEnv_dealloc);
	NativeFlappyEnvType.tp_methods = NativeFlappyEnv_methods;

	if (PyType_Ready(&NativeFlappyEnvType) < 0) {
		return nullptr;
	}

	PyObject* module = PyModule_Create(&nativeFlappyEnvModule);
	if (!module) {
		return nullptr;
	}

	Py_INCREF(&NativeFlappyEnvType);
	if (PyModule_AddObject(module, "NativeFlappyEnv", reinterpret_cast<PyObject*>(&NativeFlappyEnvType)) < 0) {
		Py_DECREF(&NativeFlappyEnvType);
		Py_DECREF(module);
		return nullptr;
	}

	return module;
}
